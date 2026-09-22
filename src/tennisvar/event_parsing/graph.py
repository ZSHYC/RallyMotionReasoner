from __future__ import annotations

from pathlib import Path
from typing import Any

from tennisvar.graph import build_graph

from .decoder import DecodedEvent


def _label(event: DecodedEvent) -> str:
    attrs = event.attributes
    values = [
        attrs.get("hitter"),
        attrs.get("court_zone"),
        attrs.get("phase"),
        attrs.get("hand"),
        attrs.get("technique"),
        attrs.get("direction"),
        "approach" if attrs.get("approach") in {"True", "true", "1", True} else "-",
        attrs.get("outcome"),
    ]
    return "_".join(str(value) if value not in {None, ""} else "-" for value in values)


def build_predicted_graph(
    events: list[DecodedEvent],
    *,
    rally_id: str,
    frames_dir: Path,
    fps: float,
    num_frames: int,
    width: int | None = None,
    height: int | None = None,
    split: str = "inference",
) -> dict[str, Any]:
    hit_events = sorted((event for event in events if event.event_type == "hit"), key=lambda event: event.frame)
    bounce_events = sorted((event for event in events if event.event_type == "bounce"), key=lambda event: event.frame)
    record = {
        "video": rally_id,
        "clip_id": rally_id,
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "events": [{"frame": event.frame, "label": _label(event), "outcome": event.attributes.get("outcome")} for event in hit_events],
        "bounce_events": [
            {"frame": event.frame, "confidence": event.confidence, "time_sec": event.time_sec}
            for event in bounce_events
        ],
        "tactical_units": [],
        "media": {"frames_dir": str(Path(frames_dir).resolve())},
    }
    graph = build_graph(record, split)
    graph["graph_source"] = "region_fusion_predicted"
    graph["event_detector"] = "region_fusion"
    graph["bounce_events"] = record["bounce_events"]
    for stroke in graph.get("strokes") or []:
        frame = int(stroke["frame"])
        before = [event for event in bounce_events if event.frame <= frame]
        after = [event for event in bounce_events if event.frame >= frame]
        if before:
            event = max(before, key=lambda item: item.frame)
            stroke["bounce_before_gap"] = frame - event.frame
            stroke["bounce_before_confidence"] = event.confidence
        if after:
            event = min(after, key=lambda item: item.frame)
            stroke["bounce_after_gap"] = event.frame - frame
            stroke["bounce_after_confidence"] = event.confidence
    for stroke, event in zip(graph.get("strokes") or [], hit_events):
        # Missing predictions stay unknown instead of becoming a negative label.
        stroke["parsed"].update(event.attributes)
        stroke["confidence"] = event.confidence
        stroke["attribute_confidence"] = event.attribute_confidence
        stroke["source"] = event.source
    return graph
