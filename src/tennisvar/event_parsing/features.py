from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VISUAL_DIM = 768
MOTION_DIM = 24
BALL_FRAME_DIM = 8
EVENT_FEATURE_DIM = VISUAL_DIM + MOTION_DIM + BALL_FRAME_DIM


@dataclass(frozen=True)
class FeatureProvenance:
    backend: str
    weights: str
    weights_sha256: str | None
    feature_dim: int
    frame_count: int
    ball_status: str
    tracknet_checkpoint_sha256: str | None = None
    tracknet_source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gray(path: Path, size: int = 64) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


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


def motion_features(paths: list[Path]) -> np.ndarray:
    if not paths:
        return np.zeros((0, MOTION_DIM), dtype=np.float32)
    import torch

    gray = torch.from_numpy(np.stack([_gray(path) for path in paths], axis=0))
    return motion_features_from_gray(gray).cpu().numpy().astype(np.float32, copy=False)


def ball_frame_features(frame_indices: list[int], track: dict[str, Any] | None) -> tuple[np.ndarray, str]:
    output = np.zeros((len(frame_indices), BALL_FRAME_DIM), dtype=np.float32)
    if not track:
        return output, "MISSING_OPTIONAL_TRACK"
    width = max(float(track.get("width") or track.get("image_width") or 1), 1.0)
    height = max(float(track.get("height") or track.get("image_height") or 1), 1.0)
    points = {int(item["frame"]): item for item in track.get("points") or [] if item.get("frame") is not None}
    previous_x = previous_y = 0.0
    visible_count = 0
    for row_index, frame in enumerate(frame_indices):
        point = points.get(int(frame))
        if not point:
            continue
        x = float(point.get("ball_x", point.get("x", 0.0))) / width
        y = float(point.get("ball_y", point.get("y", 0.0))) / height
        confidence = float(point.get("confidence", 0.0))
        visible = float(bool(point.get("visible", confidence > 0)))
        dx, dy = x - previous_x, y - previous_y
        output[row_index] = [x, y, confidence, visible, dx, dy, float(np.hypot(dx, dy)), float(frame) / max(frame_indices[-1], 1)]
        previous_x, previous_y = x, y
        visible_count += int(visible)
    return output, "READY" if visible_count else "EMPTY_OR_INVISIBLE"


class DinoMotionFeatureExtractor:
    """Frozen DINOv3 frame features plus explicit motion and optional TrackNet features."""

    def __init__(
        self,
        repo: Path,
        weights: Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
        image_size: int = 224,
        require_ball: bool = False,
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
        # Hash the immutable weight file once. Re-hashing a ~340 MB checkpoint
        # for every rally would add terabytes of avoidable I/O on the full set.
        self.weights_sha256 = _sha256(self.weights)
        self.encoder = dinov3_vitb16(pretrained=True, weights=str(self.weights)).to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.batch_size, self.image_size, self.require_ball = int(batch_size), int(image_size), bool(require_ball)
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
        indices = frame_indices or list(range(len(paths)))
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
        if self.require_ball and ball_status != "READY":
            raise RuntimeError(f"TrackNet features are required but unavailable: {ball_status}")
        features = self.torch.cat([visual_tensor, motion, self.torch.from_numpy(ball)], dim=-1)
        provenance = FeatureProvenance(
            backend="dinov3_vitb16_lvd1689m+motion24+tracknet8",
            weights=str(self.weights),
            weights_sha256=self.weights_sha256,
            feature_dim=int(features.shape[-1]),
            frame_count=len(paths),
            ball_status=ball_status,
        )
        return features, provenance
