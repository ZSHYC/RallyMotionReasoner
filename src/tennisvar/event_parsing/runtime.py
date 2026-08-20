from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tennisvar.data_fingerprint import file_sha256

from .checkpoint import load_event_checkpoint
from .decoder import DecodedEvent, PeakDecoder
from .features import EVENT_FEATURE_DIM, DinoMotionFeatureExtractor, FeatureProvenance


@dataclass
class EventRuntimeOutput:
    events: list[DecodedEvent]
    frame_indices: list[int]
    frame_features: Any
    frame_scores: list[float]
    feature_provenance: FeatureProvenance
    checkpoint_provenance: dict[str, Any]


class EventPredictor:
    def __init__(
        self,
        checkpoint: Path,
        *,
        dinov3_repo: Path,
        dinov3_weights: Path,
        device: str | None = None,
        require_ball: bool = True,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint = load_event_checkpoint(checkpoint, device=str(self.device))
        self.checkpoint = {**self.checkpoint, "checkpoint_sha256": file_sha256(Path(checkpoint))}
        run_manifest = self.checkpoint.get("run_manifest") or {}
        self.feature_contract = run_manifest.get("feature_contract") or {}
        backend = str(run_manifest.get("event_feature_backend") or "")
        self.requires_tracknet = "tracknet8_real" in backend
        decoder = self.checkpoint.get("decoder") or {}
        self.decoder = PeakDecoder(
            threshold=float(decoder.get("threshold", 0.5)),
            min_separation_frames=int(decoder.get("min_separation_frames", decoder.get("nms_frames", 8))),
            top_k=int(decoder.get("top_k", 32)),
        )
        self.extractor = DinoMotionFeatureExtractor(
            dinov3_repo,
            dinov3_weights,
            device=str(self.device),
            require_ball=require_ball,
        )
        if self.extractor.weights_sha256 != self.feature_contract.get("weights_sha256"):
            raise ValueError("DINOv3 weights hash does not match the event model training contract")
        expected_dim = int(self.checkpoint["model_config"]["input_dim"])
        if expected_dim != EVENT_FEATURE_DIM:
            raise ValueError(f"event checkpoint expects unsupported input dimension: {expected_dim} != {EVENT_FEATURE_DIM}")

    def predict(
        self,
        paths: list[Path],
        *,
        fps: float,
        frame_indices: list[int] | None = None,
        ball_track: dict[str, Any] | None = None,
    ) -> EventRuntimeOutput:
        if self.requires_tracknet and ball_track is None:
            raise ValueError("this EPM checkpoint requires a TrackNet-compatible trajectory payload")
        features, provenance = self.extractor.extract(paths, frame_indices=frame_indices, ball_track=ball_track)
        if ball_track:
            provenance = replace(
                provenance,
                backend=str(self.feature_contract.get("backend") or provenance.backend),
                tracknet_checkpoint_sha256=ball_track.get("tracker_checkpoint_sha256"),
                tracknet_source=ball_track.get("source"),
            )
        if provenance.weights_sha256 != self.feature_contract.get("weights_sha256"):
            raise RuntimeError("runtime DINOv3 provenance differs from the event checkpoint")
        if provenance.backend != self.feature_contract.get("backend"):
            raise RuntimeError("runtime event-feature backend differs from the event checkpoint")
        if self.requires_tracknet and provenance.tracknet_checkpoint_sha256 != self.feature_contract.get(
            "tracknet_checkpoint_sha256"
        ):
            raise RuntimeError("runtime TrackNet provenance differs from the event checkpoint")
        expected = int(self.checkpoint["model_config"]["input_dim"])
        if int(features.shape[-1]) != expected:
            raise ValueError(f"event feature dimension mismatch: {features.shape[-1]} != {expected}")
        frames = frame_indices or list(range(len(paths)))
        with self.torch.no_grad():
            outputs = self.model(features.unsqueeze(0).to(self.device), self.torch.ones(1, len(paths), dtype=self.torch.bool, device=self.device))
        scores = self.torch.sigmoid(outputs["event_logits"][0]).detach().cpu().tolist()
        attributes = {
            field: self.torch.softmax(outputs[f"{field}_logits"][0], dim=-1).detach().cpu().tolist()
            for field in self.checkpoint["attribute_maps"]
            if f"{field}_logits" in outputs
        }
        events = self.decoder.decode(scores, frames, attributes, self.checkpoint["attribute_maps"], fps=fps)
        return EventRuntimeOutput(events, frames, features, scores, provenance, self.checkpoint)
