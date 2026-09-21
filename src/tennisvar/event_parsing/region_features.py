"""Inputs for the tennis-region-infer event experts."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .features import DinoMotionFeatureExtractor, FeatureProvenance


def _trajectory_rows(
    track: dict[str, Any] | None,
    frame_indices: list[int],
    *,
    fps: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Build the 10-D motion contract used by the published trajectory expert."""
    rows = np.zeros((len(frame_indices), 10), dtype=np.float32)
    rows[:, 9] = 1.0
    if not track:
        return rows
    width, height = max(int(width), 1), max(int(height), 1)
    points = {int(item["frame"]): item for item in track.get("points") or [] if item.get("frame") is not None}
    previous_index = None
    previous_speed = previous_elapsed = previous_angle = None
    times = np.asarray(frame_indices, dtype=np.float64) / max(float(fps), 1e-6)
    for index, frame in enumerate(frame_indices):
        point = points.get(int(frame)) or {}
        x = point.get("ball_x", point.get("x"))
        y = point.get("ball_y", point.get("y"))
        visible = bool(point.get("visible", float(point.get("confidence", 0.0)) > 0))
        if x is None or y is None or not visible:
            continue
        x_norm = float(np.float32(float(x) / width))
        y_norm = float(np.float32(float(y) / height))
        rows[index, :3] = x_norm, y_norm, 1.0
        if previous_index is not None:
            elapsed = float(times[index] - times[previous_index])
            if 0.0 < elapsed <= 0.2:
                dx = (x_norm - float(rows[previous_index, 0])) / elapsed
                dy = (y_norm - float(rows[previous_index, 1])) / elapsed
                speed = math.hypot(dx, dy)
                derivative_elapsed = elapsed if previous_elapsed is None else 0.5 * (previous_elapsed + elapsed)
                acceleration = 0.0 if previous_speed is None else (speed - previous_speed) / derivative_elapsed
                angle = None if speed == 0.0 else math.atan2(dy, dx)
                angle_change = 0.0
                if previous_angle is not None and angle is not None:
                    angle_change = abs((angle - previous_angle + math.pi) % (2 * math.pi) - math.pi)
                mean_speed = speed if previous_speed is None else 0.5 * (previous_speed + speed)
                curvature = math.log1p((angle_change / derivative_elapsed) / max(mean_speed, 1e-6))
                rows[index, 3:9] = dx, dy, speed, acceleration, angle_change / math.pi, curvature
                previous_speed, previous_elapsed, previous_angle = speed, elapsed, angle
            else:
                previous_speed = previous_elapsed = previous_angle = None
        previous_index = index
    return rows


def normalize_trajectory(rows: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    indices = [0, 1, 3, 4, 5, 6, 7, 8]
    center = np.asarray(normalizer["center"], dtype=np.float32)
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    if center.shape != (8,) or scale.shape != (8,) or np.any(scale <= 0):
        raise ValueError("region trajectory normalizer must contain eight positive scales")
    active = (rows[:, 2] > 0.5) & (rows[:, 9] > 0.5)
    output = rows.copy()
    output[:, indices] = 0.0
    output[np.ix_(active, indices)] = np.clip(
        (rows[np.ix_(active, indices)] - center) / scale,
        -float(normalizer.get("clip", 5.0)),
        float(normalizer.get("clip", 5.0)),
    )
    return output


def _tile_arrays(array: np.ndarray) -> list[np.ndarray]:
    height, width = array.shape[:2]
    h, w = (55 * height + 99) // 100, (55 * width + 99) // 100
    return [array[:h, :w], array[:h, width - w:], array[height - h:, :w], array[height - h:, width - w:]]


class RegionFeatureExtractor:
    """Reuse TennisVAR's DINO provenance while adding four raw-image regions."""

    def __init__(self, repo: Path, weights: Path, *, device: str, batch_size: int = 32) -> None:
        import torch

        self.torch = torch
        self.base = DinoMotionFeatureExtractor(repo, weights, device=device, require_ball=True)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("region DINO batch size must be positive")

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
        from PIL import Image

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
        base_features, provenance = self.base.extract(paths, frame_indices=frame_indices, ball_track=ball_track)
        visual_features = self._region_features(paths)
        trajectory = _trajectory_rows(
            ball_track,
            frame_indices,
            fps=fps,
            width=int((ball_track or {}).get("width") or (ball_track or {}).get("image_width") or 1),
            height=int((ball_track or {}).get("height") or (ball_track or {}).get("image_height") or 1),
        )
        provenance = replace(provenance, backend=f"{provenance.backend}+region4")
        return base_features, (trajectory, visual_features), provenance
