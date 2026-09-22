from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tennisvar.features.ball_trajectory import trajectory_rows

VISUAL_DIM = 768
MOTION_DIM = 24
BALL_FRAME_DIM = 8
EVENT_FEATURE_DIM = VISUAL_DIM + MOTION_DIM + BALL_FRAME_DIM


@dataclass(frozen=True)
class FeatureProvenance:
    backend: str
    weights: str
    feature_dim: int
    frame_count: int
    ball_status: str
    tracknet_source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def shot_feature_payload(
    graph: dict[str, Any], frame_indices: list[int], frame_features: Any, *, split: str
) -> dict[str, Any]:
    """Select hit-node features for both exported datasets and live TGTR input."""
    if frame_features.ndim != 2 or len(frame_features) != len(frame_indices):
        raise ValueError("frame features and frame indices must align")
    strokes = graph.get("strokes") or []
    by_frame = {frame: index for index, frame in enumerate(frame_indices)}
    indices = [by_frame[int(shot["frame"])] for shot in strokes]
    selected = frame_features[indices].float().cpu()
    return {
        "schema": "tennisvar.tgtr_event_features",
        "source": "region_fusion_predicted",
        "rally_id": graph["rally_id"],
        "split": split,
        "shot_ids": [int(shot["shot_id"]) for shot in strokes],
        "shot_frames": [int(shot["frame"]) for shot in strokes],
        "shot_features": selected,
        "motion_stats": selected[:, VISUAL_DIM : VISUAL_DIM + MOTION_DIM],
    }


def motion_features_from_gray(gray: Any) -> Any:
    """Vectorized 24-D motion statistics for [frames, height, width] gray tensors."""
    import torch

    if gray.ndim != 3:
        raise ValueError("gray tensor must have shape [frames, height, width]")
    if not int(gray.shape[0]):
        return gray.new_zeros((0, MOTION_DIM))
    previous = torch.cat([gray[:1], gray[:-1]], dim=0)
    following = torch.cat([gray[1:], gray[-1:]], dim=0)

    def stats(value: Any) -> Any:
        flat = value.flatten(1).float()
        quantiles = torch.quantile(
            flat,
            torch.tensor([0.10, 0.25, 0.75, 0.90], device=flat.device, dtype=flat.dtype),
            dim=1,
        ).transpose(0, 1)
        return torch.cat(
            [
                flat.mean(dim=1, keepdim=True),
                flat.std(dim=1, unbiased=False, keepdim=True),
                flat.amin(dim=1, keepdim=True),
                flat.amax(dim=1, keepdim=True),
                quantiles,
            ],
            dim=1,
        )

    return torch.cat(
        [stats((gray - previous).abs()), stats((following - gray).abs()), stats((following - previous).abs())],
        dim=1,
    )


def ball_frame_features(frame_indices: list[int], track: dict[str, Any] | None) -> tuple[np.ndarray, str]:
    output = np.zeros((len(frame_indices), BALL_FRAME_DIM), dtype=np.float32)
    if not track:
        return output, "MISSING_OPTIONAL_TRACK"
    width = max(float(track.get("width") or track.get("image_width") or 1), 1.0)
    height = max(float(track.get("height") or track.get("image_height") or 1), 1.0)
    points = {int(item["frame"]): item for item in track.get("points") or [] if item.get("frame") is not None}
    previous_x = previous_y = 0.0
    previous_frame: int | None = None
    visible_count = 0
    for row_index, frame in enumerate(frame_indices):
        point = points.get(int(frame))
        if not point:
            continue
        raw_x = point.get("ball_x", point.get("x"))
        raw_y = point.get("ball_y", point.get("y"))
        confidence = float(point.get("confidence", 0.0))
        visible = float(bool(point.get("visible", confidence > 0)) and raw_x is not None and raw_y is not None)
        if not visible:
            output[row_index, 2] = max(0.0, min(1.0, confidence))
            output[row_index, 7] = float(frame - frame_indices[0]) / max(frame_indices[-1] - frame_indices[0], 1)
            continue
        x = float(raw_x) / width
        y = float(raw_y) / height
        if previous_frame is None:
            dx = dy = 0.0
        else:
            elapsed = max(int(frame) - previous_frame, 1)
            dx, dy = (x - previous_x) / elapsed, (y - previous_y) / elapsed
        output[row_index] = [
            x,
            y,
            max(0.0, min(1.0, confidence)),
            visible,
            dx,
            dy,
            float(np.hypot(dx, dy)),
            float(frame - frame_indices[0]) / max(frame_indices[-1] - frame_indices[0], 1),
        ]
        previous_x, previous_y, previous_frame = x, y, int(frame)
        visible_count += int(visible)
    return output, "READY" if visible_count else "EMPTY_OR_INVISIBLE"


class FrameFeatureExtractor:
    """Frozen DINOv3 frame features plus explicit motion and optional TrackNet features."""

    def __init__(
        self,
        repo: Path,
        weights: Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
        image_size: int = 224,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("DINO event feature extraction requires torch") from exc
        self.torch = torch
        self.repo, self.weights = Path(repo), Path(weights)
        if not self.repo.is_dir() or not self.weights.is_file():
            raise FileNotFoundError(f"missing DINOv3 assets: repo={self.repo}, weights={self.weights}")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from dinov3.hub.backbones import dinov3_vitb16

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = dinov3_vitb16(pretrained=True, weights=str(self.weights)).to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.batch_size, self.image_size = int(batch_size), int(image_size)
        if self.batch_size <= 0 or self.image_size <= 0:
            raise ValueError("DINO batch size and image size must be positive")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.mean_device = self.mean.to(self.device)
        self.std_device = self.std.to(self.device)
        self.gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=self.device).view(1, 3, 1, 1)

    def _tensor(self, path: Path):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR), dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std

    def extract(
        self,
        paths: list[Path],
        *,
        frame_indices: list[int] | None = None,
        ball_track: dict[str, Any] | None = None,
    ) -> tuple[Any, FeatureProvenance]:
        if not paths:
            raise ValueError("cannot extract event features from an empty frame sequence")
        indices = list(range(len(paths))) if frame_indices is None else frame_indices
        if len(indices) != len(paths):
            raise ValueError("frame_indices and paths length mismatch")
        import torch.nn.functional as functional

        visual = []
        gray_frames = []
        with self.torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch = self.torch.stack([self._tensor(path) for path in paths[start : start + self.batch_size]]).to(self.device)
                rgb = batch * self.std_device.unsqueeze(0) + self.mean_device.unsqueeze(0)
                gray = (rgb * self.gray_weights).sum(dim=1, keepdim=True)
                gray_frames.append(
                    functional.interpolate(gray, size=(64, 64), mode="bilinear", align_corners=False).squeeze(1)
                )
                encoded = self.encoder(batch)
                if isinstance(encoded, dict):
                    raw_output = encoded
                    encoded = raw_output.get("x_norm_clstoken")
                    if encoded is None:
                        encoded = raw_output.get("last_hidden_state")
                    if encoded is None:
                        raise RuntimeError("DINOv3 encoder returned no supported feature tensor")
                if encoded.ndim == 3:
                    encoded = encoded[:, 0]
                visual.append(encoded.detach().float().cpu())
        visual_tensor = self.torch.cat(visual, dim=0)
        if visual_tensor.shape[-1] != VISUAL_DIM:
            raise RuntimeError(f"unexpected DINOv3 output dimension: {visual_tensor.shape[-1]} != {VISUAL_DIM}")
        motion = motion_features_from_gray(self.torch.cat(gray_frames, dim=0)).detach().float().cpu()
        ball, ball_status = ball_frame_features(indices, ball_track)
        features = self.torch.cat([visual_tensor, motion, self.torch.from_numpy(ball)], dim=-1)
        provenance = FeatureProvenance(
            backend="dinov3_vitb16_lvd1689m+motion24+tracknet8",
            weights=str(self.weights),
            feature_dim=int(features.shape[-1]),
            frame_count=len(paths),
            ball_status=ball_status,
            tracknet_source=(ball_track or {}).get("source"),
        )
        return features, provenance


def _tile_arrays(array: np.ndarray) -> list[np.ndarray]:
    height, width = array.shape[:2]
    h, w = (55 * height + 99) // 100, (55 * width + 99) // 100
    return [array[:h, :w], array[:h, width - w:], array[height - h:, :w], array[height - h:, width - w:]]


class RegionFeatureExtractor:
    """Reuse TennisVAR's DINO provenance while adding four raw-image regions."""

    def __init__(self, repo: Path, weights: Path, *, device: str, batch_size: int = 32) -> None:
        import torch

        self.torch = torch
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("region DINO batch size must be positive")
        self.base = FrameFeatureExtractor(repo, weights, device=device, batch_size=self.batch_size)

    def _image_tensor(self, array: np.ndarray) -> Any:
        import torch.nn.functional as functional

        tensor = self.torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
        tensor = functional.interpolate(
            tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False, antialias=True
        )[0]
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    def _encode(self, batch: Any) -> Any:
        encoded = self.base.encoder(batch)
        if isinstance(encoded, dict):
            raw_output = encoded
            encoded = raw_output.get("x_norm_clstoken")
            if encoded is None:
                encoded = raw_output.get("last_hidden_state")
            if encoded is None:
                raise RuntimeError("DINOv3 encoder returned no CLS feature")
        if encoded.ndim == 3:
            encoded = encoded[:, 0]
        return encoded

    def _region_features(self, paths: list[Path]) -> Any:
        values: list[Any] = []
        with self.torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch_arrays: list[np.ndarray] = []
                for path in paths[start : start + self.batch_size]:
                    with Image.open(path) as image:
                        batch_arrays.append(np.asarray(image.convert("RGB")))
                views = [self._image_tensor(array) for array in batch_arrays]
                views.extend(self._image_tensor(tile) for array in batch_arrays for tile in _tile_arrays(array))
                batch = self.torch.stack(views).to(self.base.device)
                encoded = self._encode(batch).detach().float().cpu()
                if encoded.ndim != 2 or encoded.shape[-1] != 768 or not self.torch.isfinite(encoded).all():
                    raise RuntimeError("region DINO encoder must return finite [N,768] features")
                full = encoded[: len(batch_arrays)]
                tiles = encoded[len(batch_arrays) :].reshape(len(batch_arrays), 4, 768)
                values.append(self.torch.cat((full.unsqueeze(1), tiles), dim=1))
        return self.torch.cat(values, dim=0).reshape(len(paths), 3840).to(dtype=self.torch.float16)

    def extract(
        self,
        paths: list[Path],
        *,
        frame_indices: list[int],
        fps: float,
        ball_track: dict[str, Any] | None,
    ) -> tuple[Any, Any, FeatureProvenance]:
        if len(paths) != len(frame_indices) or not paths:
            raise ValueError("region paths and frame_indices must be non-empty and aligned")
        if ball_track is None:
            raise ValueError("region event detection requires a trajectory payload")
        base_features, provenance = self.base.extract(paths, frame_indices=frame_indices, ball_track=ball_track)
        visual_features = self._region_features(paths)
        trajectory = trajectory_rows(
            ball_track,
            frame_indices,
            fps=fps,
            width=int((ball_track or {}).get("width") or (ball_track or {}).get("image_width") or 1),
            height=int((ball_track or {}).get("height") or (ball_track or {}).get("image_height") or 1),
        )
        provenance = replace(provenance, backend=f"{provenance.backend}+region4")
        return base_features, (trajectory, visual_features), provenance
