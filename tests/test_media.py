import json
from types import SimpleNamespace

import pytest

from rallymotionreasoner.media import _require_constant_fps, materialize_video


def test_video_timing_contract_rejects_variable_rate_and_unknown_directory_fps(tmp_path, monkeypatch) -> None:
    def ffprobe(_command, **_kwargs):
        return SimpleNamespace(stdout=json.dumps({
            "streams": [{"time_base": "1/25"}],
            "packets": [{"pts": value} for value in ffprobe.pts],
        }))

    monkeypatch.setattr("rallymotionreasoner.media.subprocess.run", ffprobe)
    ffprobe.pts = [0, 1, 2]
    _require_constant_fps(tmp_path / "video.mp4", 3, 25.0)
    ffprobe.pts = [0, 1, 3]
    with pytest.raises(ValueError, match="variable-frame-rate"):
        _require_constant_fps(tmp_path / "video.mp4", 3, 25.0)

    (tmp_path / "frame_000000.png").touch()
    with pytest.raises(ValueError, match="explicit positive FPS"):
        with materialize_video(tmp_path):
            pass
