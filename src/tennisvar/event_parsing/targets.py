from __future__ import annotations

import math
from collections.abc import Iterable


def event_heatmap(frame_indices: list[int], event_frames: Iterable[int], radius: int = 4, mode: str = "gaussian") -> list[float]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    events = [int(item) for item in event_frames]
    sigma = max(radius / 2.0, 1.0)
    values: list[float] = []
    for frame in frame_indices:
        best = 0.0
        for event in events:
            distance = abs(int(frame) - event)
            if mode == "triangular":
                score = max(0.0, 1.0 - distance / max(radius, 1)) if radius else float(distance == 0)
            elif mode == "gaussian":
                score = math.exp(-0.5 * (distance / sigma) ** 2) if distance <= radius else 0.0
            else:
                raise ValueError(f"unknown heatmap mode: {mode}")
            best = max(best, score)
        values.append(float(best))
    return values


def binary_event_labels(heatmap: list[float], threshold: float = 0.05) -> list[int]:
    return [int(float(value) > threshold) for value in heatmap]


def hard_negative_mask(frame_indices: list[int], event_frames: Iterable[int], exclusion_radius: int = 8) -> list[int]:
    events = [int(item) for item in event_frames]
    return [int(not any(abs(int(frame) - event) <= exclusion_radius for event in events)) for frame in frame_indices]
