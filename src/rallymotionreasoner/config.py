from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "paths.local.yaml"

DEFAULT_PATHS = {
    "data_root": "data",
    "source_data_root": "data/source/trace",
    "frame_root": "data/media/frames_224",
    "video_root": "data/media/videos",
    "artifacts_root": "artifacts",
    "reports_root": "reports",
    "runs_root": "runs",
    "outputs_root": "outputs",
    "qwen3_vl_model": "model_zoo/Qwen3-VL-8B-Instruct",
    "dinov3_repo": "model_zoo/dinov3",
    "dinov3_weights": "model_zoo/dinov3_vitb16.pth",
    "ball_track_root": "artifacts/ball_tracks",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"path configuration must be a mapping: {path}")
    return payload


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_paths(path: Path | None = None) -> dict[str, Path]:
    """Load machine-local paths without embedding them in tracked files."""

    config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    raw = {**DEFAULT_PATHS, **_read_yaml(config_path)}
    project_value = os.environ.get("RALLYMOTIONREASONER_PROJECT_ROOT", raw.pop("project_root", REPOSITORY_ROOT))
    project_root = _resolve(project_value, base=REPOSITORY_ROOT)
    resolved: dict[str, Path] = {"project_root": project_root}
    for key, default in raw.items():
        value = os.environ.get(f"RALLYMOTIONREASONER_{key.upper()}", default)
        resolved[key] = _resolve(value, base=project_root)
    return resolved
