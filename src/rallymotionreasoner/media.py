from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from rallymotionreasoner.video import list_images


@dataclass(frozen=True)
class MaterializedVideo:
    frame_paths: list[Path]
    fps: float
    source: str
    decoder: str
    source_size: tuple[int, int] | None = None


def _require_constant_fps(path: Path, frame_count: int, fps: float) -> None:
    """The event and Qwen contracts both encode time as frame index / FPS."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=time_base:packet=pts", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout)
        streams, packets = payload["streams"], payload["packets"]
        scale = float(Fraction(streams[0]["time_base"]))
        times = sorted(int(packet["pts"]) * scale for packet in packets)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise ValueError(f"ffprobe could not establish video frame timing: {path}") from exc
    if len(streams) != 1 or len(times) != frame_count or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"video frame timing does not match decoded frames: {path}")
    tolerance = max(0.001, 0.05 / fps)
    if any(not math.isfinite(time) for time in times) or any(
        abs(right - left - 1.0 / fps) > tolerance for left, right in zip(times, times[1:])
    ):
        raise ValueError(f"variable-frame-rate video is unsupported by the frame-index/FPS timing contract: {path}")


@contextmanager
def materialize_video(value: str | Path, *, fps: float | None = None) -> Iterator[MaterializedVideo]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        frames = list_images(path)
        if not frames:
            raise ValueError(f"frame directory is empty: {path}")
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError("frame directories require an explicit positive FPS")
        yield MaterializedVideo(frames, float(fps), str(path), "frame_directory")
        return
    if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError(f"unsupported video input: {path}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("raw video input requires the 'video' extra (opencv-python-headless)") from exc
    with tempfile.TemporaryDirectory(prefix="rallymotionreasoner_frames_") as temp:
        output = Path(temp)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 0.0)
        if not math.isfinite(detected_fps) or detected_fps <= 0:
            capture.release()
            raise ValueError(f"video has no valid FPS: {path}")
        source_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if min(source_size) <= 0:
            capture.release()
            raise ValueError(f"video has no valid frame dimensions: {path}")
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            target = output / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(target), frame):
                capture.release()
                raise RuntimeError(f"failed to write decoded frame: {target}")
            index += 1
        capture.release()
        frames = list_images(output)
        if not frames:
            raise RuntimeError(f"video decoded to zero frames: {path}")
        _require_constant_fps(path, len(frames), detected_fps)
        yield MaterializedVideo(frames, detected_fps, str(path), "opencv", source_size)
