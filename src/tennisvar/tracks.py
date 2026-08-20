"""Canonical generator-training track used by TennisVAR."""

TRACK_TGTR_ASSISTED_PRED = "tgtr_assisted_pred"


def normalize_track(track: str | None) -> str:
    value = str(track or TRACK_TGTR_ASSISTED_PRED)
    if value != TRACK_TGTR_ASSISTED_PRED:
        raise ValueError(f"unknown training track: {track}")
    return value
