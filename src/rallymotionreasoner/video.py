from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_media_path(data_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(data_root) / path


def list_images(clip_dir: Path) -> list[Path]:
    if not clip_dir.is_dir():
        return []
    return sorted(p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if k == 1:
        return [n // 2]
    if n >= k:
        return [min(n - 1, round(i * (n - 1) / (k - 1))) for i in range(k)]
    out = list(range(n))
    while len(out) < k:
        out.append(out[-1])
    return out[:k]


def local_indices(center: int, n: int, radius: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    lo = max(0, center - radius)
    hi = min(n - 1, center + radius)
    window = list(range(lo, hi + 1))
    picks = uniform_indices(len(window), k)
    return [window[i] for i in picks]


def sample_frame_paths(
    data_root: Path,
    graph: dict[str, Any],
    evidence_shot_ids: list[int] | None = None,
    global_frames: int = 8,
    local_frames: int = 4,
    local_radius: int = 8,
) -> dict[str, Any]:
    media = graph.get("media") or {}
    rel = media.get("frames_dir") or f"media/frames_224/{graph.get('clip_id')}"
    clip_dir = resolve_media_path(data_root, rel)
    images = list_images(clip_dir)
    global_paths = [str(images[i]) for i in uniform_indices(len(images), global_frames)]
    by_id = {int(s["shot_id"]): s for s in graph.get("strokes") or []}
    local: dict[str, list[str]] = {}
    for sid in evidence_shot_ids or []:
        shot = by_id.get(int(sid))
        if not shot or not images:
            continue
        center = max(0, min(len(images) - 1, int(shot["frame"])))
        local[f"SHOT_{sid}"] = [str(images[i]) for i in local_indices(center, len(images), local_radius, local_frames)]
    return {
        "clip_dir": str(clip_dir),
        "clip_dir_exists": clip_dir.is_dir(),
        "num_available_frames": len(images),
        "sampling": f"global_{global_frames}_local_{local_frames}_radius_{local_radius}",
        "evidence_shot_ids": [int(sid) for sid in evidence_shot_ids or []],
        "global_context_frames": global_paths,
        "stroke_local_frames": local,
    }
