"""Canonical generator-training track used by RallyMotionReasoner."""

TRACK_GRAPH_ASSISTED_PRED = "graph_reasoner_assisted_pred"


def normalize_track(track: str | None) -> str:
    value = str(track or TRACK_GRAPH_ASSISTED_PRED)
    if value != TRACK_GRAPH_ASSISTED_PRED:
        raise ValueError(f"unknown training track: {track}")
    return value
