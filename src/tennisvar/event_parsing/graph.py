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
    record = {
        "video": rally_id,
        "clip_id": rally_id,
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "events": [{"frame": event.frame, "label": _label(event), "outcome": event.attributes.get("outcome")} for event in events],
        "tactical_units": [],
        "media": {"frames_dir": str(Path(frames_dir).resolve())},
    }
    graph = build_graph(record, split)
    graph["graph_source"] = "f3ed_predicted"
    for stroke, event in zip(graph.get("strokes") or [], events):
        stroke["confidence"] = event.confidence
        stroke["attribute_confidence"] = event.attribute_confidence
        stroke["source"] = event.source
    return graph
