"""TrackNet input and the region expert motion contract."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_track_payload(path: Path) -> dict[str, Any]:
    """Load RallyMotionReasoner JSON tracks or the external TrackNet CSV contract."""
    if path.suffix.lower() != ".csv":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ball track JSON must be an object")
        return payload
    points: list[dict[str, Any]] = []
    width = height = None
    previous_frame = -1
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_number", "detected", "x_orig", "y_orig", "width", "height"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"TrackNet CSV is missing fields: {missing}: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame_number"])
                detected = int(row["detected"])
                row_width, row_height = int(row["width"]), int(row["height"])
                x = float(row["x_orig"]) if detected else None
                y = float(row["y_orig"]) if detected else None
                confidence = float(row.get("conf") or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid TrackNet CSV row: {path}:{line}") from exc
            if detected not in {0, 1} or row_width <= 0 or row_height <= 0:
                raise ValueError(f"invalid TrackNet CSV geometry/detection: {path}:{line}")
            if frame < 0 or frame <= previous_frame:
                raise ValueError(f"TrackNet CSV frames must be strictly increasing: {path}:{line}")
            if detected and (x is None or y is None or not (0.0 <= x < row_width and 0.0 <= y < row_height)):
                raise ValueError(f"TrackNet CSV coordinates are invalid: {path}:{line}")
            if not all(math.isfinite(value) for value in (confidence, *(value for value in (x, y) if value is not None))):
                raise ValueError(f"TrackNet CSV values must be finite: {path}:{line}")
            if width is None:
                width, height = row_width, row_height
            if (row_width, row_height) != (width, height):
                raise ValueError(f"TrackNet CSV changes resolution: {path}:{line}")
            points.append({"frame": frame, "ball_x": x, "ball_y": y, "confidence": confidence, "visible": bool(detected)})
            previous_frame = frame
    if not points:
        raise ValueError(f"TrackNet CSV is empty: {path}")
    return {"width": width, "height": height, "points": points, "source": "tracknet_csv"}


def trajectory_rows(
    track: dict[str, Any] | None,
    frame_indices: list[int],
    *,
    fps: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Build the 10-D motion contract used by the published trajectory expert."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("trajectory fps must be finite and positive")
    rows = np.zeros((len(frame_indices), 10), dtype=np.float32)
    # Valid describes a real temporal slot; detected (column 2) describes visibility.
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
    if center.shape != (8,) or scale.shape != (8,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
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
