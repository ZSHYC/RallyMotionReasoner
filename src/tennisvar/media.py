from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tennisvar.video import list_images


@dataclass(frozen=True)
class MaterializedVideo:
    frame_paths: list[Path]
    fps: float
    source: str
    decoder: str


@contextmanager
def materialize_video(value: str | Path, *, fps: float | None = None) -> Iterator[MaterializedVideo]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        frames = list_images(path)
        if not frames:
            raise ValueError(f"frame directory is empty: {path}")
        yield MaterializedVideo(frames, float(fps or 25.0), str(path), "frame_directory")
        return
    if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError(f"unsupported video input: {path}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("raw video input requires the 'video' extra (opencv-python-headless)") from exc
    with tempfile.TemporaryDirectory(prefix="tennisvar_frames_") as temp:
        output = Path(temp)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 25.0)
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            target = output / f"frame_{index:06d}.jpg"
            if not cv2.imwrite(str(target), frame):
                capture.release()
                raise RuntimeError(f"failed to write decoded frame: {target}")
            index += 1
        capture.release()
        frames = list_images(output)
        if not frames:
            raise RuntimeError(f"video decoded to zero frames: {path}")
        yield MaterializedVideo(frames, detected_fps, str(path), "opencv")
