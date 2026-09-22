"""Runtime for the trajectory plus region visual event detector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rallymotionreasoner.features.ball_trajectory import normalize_trajectory

from .decoder import DecodedEvent
from .features import FeatureProvenance, RegionFeatureExtractor
from .labels import ATTRIBUTE_FIELDS
from .model import TrajectoryExpert, VisualExpert


@dataclass
class EventRuntimeOutput:
    events: list[DecodedEvent]
    frame_indices: list[int]
    frame_features: Any
    frame_scores: list[float]
    feature_provenance: FeatureProvenance
    checkpoint_provenance: dict[str, Any]


EVENT_TYPES = ("hit", "bounce")


def _contract(payload: dict[str, Any], *, kind: str, radius: int, span: float, dimension: int) -> dict[str, Any]:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("region expert is missing its contract")
    expected = {
        "model_kind": kind,
        "trajectory_dim": 11,
        "visual_dim": dimension,
        "window_radius": radius,
        "window_span_seconds": span,
        "event_types": list(EVENT_TYPES),
        "score_mode": "product",
        "nms_radius": 5,
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"region expert contract mismatch for {field}: {contract.get(field)!r} != {value!r}")
    if contract.get("feature_names") != [
        "x_norm", "y_norm", "detected", "dx", "dy", "speed", "acceleration",
        "angle_change", "curvature", "valid", "time_offset",
    ]:
        raise ValueError("region expert feature names do not match the motion contract")
    return contract


def _window_indices(times: Any, centers: Any, offsets: Any) -> tuple[Any, Any]:
    import torch

    targets = times[centers, None] + offsets[None, :]
    right = times.searchsorted(targets).clamp(max=len(times) - 1)
    left = (right - 1).clamp(min=0)
    indices = torch.where(
        (times[right] - targets).abs() < (times[left] - targets).abs(), right, left
    )
    tolerance = torch.finfo(times.dtype).eps * max(
        1.0, float(times.abs().max()), float(offsets.abs().max())
    ) * 4
    valid = (targets >= times[0] - tolerance) & (targets <= times[-1] + tolerance)
    return indices, valid


def _scores(model: Any, rows: Any, times: Any, contract: dict[str, Any], device: Any, *, trajectory: bool) -> Any:
    import torch

    offsets = torch.linspace(
        -float(contract["window_span_seconds"]) / 2,
        float(contract["window_span_seconds"]) / 2,
        2 * int(contract["window_radius"]) + 1,
        dtype=torch.float64,
    )
    outputs: list[Any] = []
    for start in range(0, len(times), 512):
        centers = torch.arange(start, min(start + 512, len(times)), dtype=torch.long)
        indices, valid = _window_indices(times, centers, offsets)
        batch = rows[indices] * valid[..., None]
        if trajectory:
            relative = ((times[indices] - times[centers, None]) * valid).float()
            batch = torch.cat((batch, relative[..., None]), dim=-1)
        with torch.inference_mode():
            result = model(batch.to(device=device, dtype=torch.float32))
        outputs.append(
            result["eventness_logit"].sigmoid()[:, None] * result["type_logits"].softmax(dim=-1)
        )
    return torch.cat(outputs, dim=0).float().cpu()


def _decode(scores: Any, frames: list[int], fps: float, threshold: float, radius: int) -> list[DecodedEvent]:
    selected: list[tuple[int, int, float]] = []
    for column in range(len(EVENT_TYPES)):
        kept: list[int] = []
        candidates = sorted(
            ((int(frame), float(scores[index, column])) for index, frame in enumerate(frames)),
            key=lambda item: (-item[1], item[0]),
        )
        for frame, score in candidates:
            if score < threshold or any(abs(frame - previous) <= radius for previous in kept):
                continue
            kept.append(frame)
            selected.append((frame, column, score))
    selected.sort(key=lambda item: (item[0], EVENT_TYPES[item[1]]))
    empty_attributes = {field: None for field in ATTRIBUTE_FIELDS}
    empty_confidence = {field: 0.0 for field in ATTRIBUTE_FIELDS}
    return [
        DecodedEvent(
            frame=frame,
            time_sec=round(frame / max(float(fps), 1e-6), 4),
            confidence=score,
            attributes=dict(empty_attributes),
            attribute_confidence=dict(empty_confidence),
            source="motion_region",
            event_type=EVENT_TYPES[column],
        )
        for frame, column, score in selected
    ]


class EventPredictor:
    """Runtime for the published trajectory/visual expert directory."""

    backend = "motion_region"

    def __init__(
        self,
        model_dir: Path,
        *,
        dinov3_repo: Path,
        dinov3_weights: Path,
        device: str | None = None,
        dino_batch_size: int = 32,
        score_threshold: float = 0.4,
    ) -> None:
        import torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        trajectory_payload = torch.load(self.model_dir / "trajectory_expert.pt", map_location="cpu", weights_only=True)
        visual_payload = torch.load(self.model_dir / "visual_expert.pt", map_location="cpu", weights_only=True)
        self.trajectory_contract = _contract(
            trajectory_payload, kind="trajectory", radius=12, span=0.4, dimension=768
        )
        self.visual_contract = _contract(
            visual_payload, kind="visual_tile_region_temporal", radius=24, span=1.6, dimension=3840
        )
        normalizer = trajectory_payload.get("normalizer")
        if not isinstance(normalizer, dict) or normalizer.get("columns") != [
            "x_norm", "y_norm", "dx", "dy", "speed", "acceleration", "angle_change", "curvature"
        ]:
            raise ValueError("trajectory expert normalizer does not match the published eight-feature contract")
        self.normalizer = normalizer
        self.trajectory_model = TrajectoryExpert().to(self.device)
        self.visual_model = VisualExpert().to(self.device)
        self.trajectory_model.load_state_dict(trajectory_payload["model_state"], strict=True)
        self.visual_model.load_state_dict(visual_payload["model_state"], strict=True)
        self.trajectory_model.eval().requires_grad_(False)
        self.visual_model.eval().requires_grad_(False)
        self.extractor = RegionFeatureExtractor(
            Path(dinov3_repo), Path(dinov3_weights), device=str(self.device), batch_size=dino_batch_size
        )
        self.score_threshold = float(score_threshold)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("region event score threshold must be in [0, 1]")
        if self.trajectory_contract.get("base_spec", {}).get("coordinate_precision") not in {
            None,
            "consistent_float32",
        }:
            raise ValueError("trajectory expert requires consistent_float32 coordinates")
        self.checkpoint = {
            "schema": "rallymotionreasoner.motion_region.v1",
            "backend": self.backend,
            "expert_directory": str(self.model_dir),
            "trajectory_contract": self.trajectory_contract,
            "visual_contract": self.visual_contract,
        }

    def predict(
        self,
        paths: list[Path],
        *,
        fps: float,
        frame_indices: list[int] | None = None,
        ball_track: dict[str, Any] | None = None,
    ) -> EventRuntimeOutput:
        import torch

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("event fps must be finite and positive")
        if ball_track is None:
            raise ValueError("region event detection requires a TrackNet-compatible trajectory payload")
        frames = [int(value) for value in (range(len(paths)) if frame_indices is None else frame_indices)]
        if len(frames) != len(paths) or not paths:
            raise ValueError("region event paths and frame_indices must be non-empty and aligned")
        if any(right <= left for left, right in zip(frames, frames[1:])):
            raise ValueError("region event frame_indices must be strictly increasing")
        base_features, (trajectory, visual), provenance = self.extractor.extract(
            paths, frame_indices=frames, fps=fps, ball_track=ball_track
        )
        trajectory_tensor = torch.from_numpy(normalize_trajectory(trajectory, self.normalizer))
        visual_tensor = visual.float()
        times = torch.tensor([frame / max(float(fps), 1e-6) for frame in frames], dtype=torch.float64)
        trajectory_scores = _scores(
            self.trajectory_model,
            trajectory_tensor,
            times,
            self.trajectory_contract,
            self.device,
            trajectory=True,
        )
        visual_scores = _scores(
            self.visual_model,
            visual_tensor,
            times,
            self.visual_contract,
            self.device,
            trajectory=False,
        )
        fused = 0.5 * trajectory_scores + 0.5 * visual_scores
        events = _decode(fused, frames, fps, self.score_threshold, int(self.visual_contract["nms_radius"]))
        return EventRuntimeOutput(
            events=events,
            frame_indices=frames,
            frame_features=base_features,
            frame_scores=fused.max(dim=1).values.tolist(),
            feature_provenance=provenance,
            checkpoint_provenance=self.checkpoint,
        )
