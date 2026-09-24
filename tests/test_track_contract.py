import math

import pytest

from rallymotionreasoner.event_detection.runtime import EventPredictor
from rallymotionreasoner.features.ball_trajectory import validate_track_payload


def test_track_contract_rejects_invalid_json_and_direct_inputs() -> None:
    track = {
        "width": 100,
        "height": 50,
        "points": [{"frame": 0, "ball_x": 10.0, "ball_y": 20.0, "visible": True}],
    }
    validate_track_payload(track, [0, 1], width=100, height=50)

    cases = [
        ({**track, "width": 200}, "width"),
        ({**track, "points": track["points"] * 2}, "unique"),
        ({**track, "points": [{**track["points"][0], "ball_x": math.nan}]}, "finite"),
        ({**track, "points": [{**track["points"][0], "frame": 2}]}, "within"),
        ({**track, "points": [{**track["points"][0], "ball_x": 100.0}]}, "exceed"),
    ]
    for invalid, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_track_payload(invalid, [0, 1], width=100, height=50)


def test_event_entry_validates_track_before_feature_extraction(tmp_path) -> None:
    predictor = EventPredictor.__new__(EventPredictor)
    track = {
        "width": 100, "height": 50,
        "points": [{"frame": 0, "ball_x": math.nan, "ball_y": 20, "visible": True}],
    }
    with pytest.raises(ValueError, match="finite"):
        predictor.predict([tmp_path / "frame.png"], fps=25.0, ball_track=track, video_size=(100, 50))
