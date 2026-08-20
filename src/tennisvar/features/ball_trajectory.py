from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BALL_TRAJECTORY_FEATURE_DIM = 64


@dataclass(frozen=True)
class BallPoint:
    frame: int
    ball_x: float | None
    ball_y: float | None
    confidence: float
    visible: bool
    source: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def safe_feature_name(rally_id: str) -> str:
    return rally_id.replace("/", "__")


def ball_track_file(track_root: Path, split: str, rally_id: str) -> Path:
    return Path(track_root) / split / f"{safe_feature_name(str(rally_id))}.json"


def normalize_ball_point(row: dict[str, Any], *, source: str = "tracknet") -> BallPoint:
    frame = int(row.get("frame"))
    x = row.get("ball_x", row.get("x"))
    y = row.get("ball_y", row.get("y"))
    confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
    visible = bool(row.get("visible", confidence > 0.0))
    return BallPoint(
        frame=frame,
        ball_x=None if x is None else float(x),
        ball_y=None if y is None else float(y),
        confidence=confidence,
        visible=visible,
        source=str(row.get("source") or source),
    )


def validate_ball_point(point: BallPoint, *, width: int | None = None, height: int | None = None) -> list[str]:
    errors: list[str] = []
    if point.frame < 0:
        errors.append("negative_frame")
    if not 0.0 <= float(point.confidence) <= 1.0:
        errors.append("confidence_out_of_range")
    if point.visible and (point.ball_x is None or point.ball_y is None):
        errors.append("visible_without_coordinates")
    if point.ball_x is not None:
        if not math.isfinite(point.ball_x):
            errors.append("ball_x_not_finite")
        if width is not None and not 0.0 <= point.ball_x <= max(0, width - 1):
            errors.append("ball_x_out_of_bounds")
    if point.ball_y is not None:
        if not math.isfinite(point.ball_y):
            errors.append("ball_y_not_finite")
        if height is not None and not 0.0 <= point.ball_y <= max(0, height - 1):
            errors.append("ball_y_out_of_bounds")
    return errors


def normalize_track_rows(rows: list[dict[str, Any]], *, source: str = "tracknet") -> list[BallPoint]:
    points: list[BallPoint] = []
    seen: set[int] = set()
    for row in rows:
        point = normalize_ball_point(row, source=source)
        if point.frame in seen:
            continue
        seen.add(point.frame)
        points.append(point)
    return sorted(points, key=lambda item: item.frame)


def load_ball_track(track_root: Path | None, split: str, rally_id: str) -> dict[str, Any] | None:
    if track_root is None:
        return None
    path = ball_track_file(Path(track_root), split, rally_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: float | None) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _norm_point(point: BallPoint, width: int, height: int) -> tuple[float, float]:
    return (
        _finite(point.ball_x) / max(width - 1, 1),
        _finite(point.ball_y) / max(height - 1, 1),
    )


def _zone_one_hot(x: float, y: float) -> list[float]:
    col = min(2, max(0, int(x * 3.0)))
    row = min(2, max(0, int(y * 3.0)))
    out = [0.0] * 9
    out[row * 3 + col] = 1.0
    return out


def _trajectory_deltas(coords: list[tuple[float, float]]) -> tuple[list[float], list[float], list[float]]:
    velocities: list[float] = []
    accelerations: list[float] = []
    curvatures: list[float] = []
    for a, b in zip(coords, coords[1:]):
        velocities.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    for a, b in zip(velocities, velocities[1:]):
        accelerations.append(abs(b - a))
    for a, b, c in zip(coords, coords[1:], coords[2:]):
        curvatures.append(math.hypot(c[0] - 2.0 * b[0] + a[0], c[1] - 2.0 * b[1] + a[1]))
    return velocities, accelerations, curvatures


def trajectory_feature_for_window(
    points: list[BallPoint],
    *,
    center_frame: int,
    radius: int,
    width: int = 224,
    height: int = 224,
) -> tuple[list[float], dict[str, Any]]:
    lo = int(center_frame) - int(radius)
    hi = int(center_frame) + int(radius)
    window = [p for p in points if lo <= int(p.frame) <= hi]
    visible = [p for p in window if p.visible and p.ball_x is not None and p.ball_y is not None]
    coords = [_norm_point(p, width, height) for p in visible]
    confidences = [float(p.confidence) for p in window]
    visible_ratio = len(visible) / max(len(window), 1)
    mean_conf = _mean(confidences)
    if coords:
        mean_x = _mean([x for x, _ in coords])
        mean_y = _mean([y for _, y in coords])
        start_x, start_y = coords[0]
        end_x, end_y = coords[-1]
    else:
        mean_x = mean_y = start_x = start_y = end_x = end_y = 0.0
    velocities, accelerations, curvatures = _trajectory_deltas(coords)
    pre = [p for p in visible if p.frame <= center_frame]
    post = [p for p in visible if p.frame >= center_frame]
    pre_coords = [_norm_point(p, width, height) for p in pre]
    post_coords = [_norm_point(p, width, height) for p in post]
    pre_dx = pre_coords[-1][0] - pre_coords[0][0] if len(pre_coords) >= 2 else 0.0
    pre_dy = pre_coords[-1][1] - pre_coords[0][1] if len(pre_coords) >= 2 else 0.0
    post_dx = post_coords[-1][0] - post_coords[0][0] if len(post_coords) >= 2 else 0.0
    post_dy = post_coords[-1][1] - post_coords[0][1] if len(post_coords) >= 2 else 0.0
    nearest = min(visible, key=lambda p: abs(int(p.frame) - int(center_frame))) if visible else None
    contact_rel = 0.0 if nearest is None else (float(nearest.frame) - float(center_frame)) / max(radius, 1)
    contact_conf = float(nearest.confidence) if nearest is not None else 0.0
    landing_x, landing_y = (coords[-1] if coords else (0.0, 0.0))
    feature = [
        visible_ratio,
        mean_conf,
        mean_x,
        mean_y,
        start_x,
        start_y,
        end_x,
        end_y,
        end_x - start_x,
        end_y - start_y,
        _mean(velocities),
        _max(velocities),
        _mean(accelerations),
        _max(accelerations),
        _mean(curvatures),
        _max(curvatures),
        pre_dx,
        pre_dy,
        post_dx,
        post_dy,
        contact_rel,
        contact_conf,
        landing_x,
        landing_y,
    ]
    feature.extend(_zone_one_hot(landing_x, landing_y))
    feature.extend([float(len(window)), float(len(visible)), float(radius), float(width), float(height)])
    if len(feature) < BALL_TRAJECTORY_FEATURE_DIM:
        feature.extend([0.0] * (BALL_TRAJECTORY_FEATURE_DIM - len(feature)))
    metadata = {
        "window_start": lo,
        "window_end": hi,
        "num_points": len(window),
        "num_visible": len(visible),
        "visible_ratio": visible_ratio,
        "mean_confidence": mean_conf,
        "status": "READY" if visible else "EMPTY_OR_INVISIBLE",
    }
    return feature[:BALL_TRAJECTORY_FEATURE_DIM], metadata


def graph_ball_features(
    track_payload: dict[str, Any] | None,
    graph: dict[str, Any],
    *,
    radius: int,
) -> tuple[list[list[float]], list[dict[str, Any]], str]:
    strokes = graph.get("strokes") or []
    if not track_payload:
        return [[0.0] * BALL_TRAJECTORY_FEATURE_DIM for _ in strokes], [{"status": "MISSING_TRACK"} for _ in strokes], "MISSING_TRACK"
    width = int(track_payload.get("image_width") or track_payload.get("width") or 224)
    height = int(track_payload.get("image_height") or track_payload.get("height") or 224)
    points = normalize_track_rows(list(track_payload.get("points") or []), source=str(track_payload.get("source") or "tracknet"))
    features: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    for idx, shot in enumerate(strokes):
        center = int(shot.get("frame") or 0)
        feature, meta = trajectory_feature_for_window(points, center_frame=center, radius=radius, width=width, height=height)
        meta["shot_id"] = int(shot.get("shot_id", idx + 1))
        meta["center_frame"] = center
        features.append(feature)
        metadata.append(meta)
    status = "READY" if any(row.get("status") == "READY" for row in metadata) else "EMPTY_OR_INVISIBLE"
    return features, metadata, status
