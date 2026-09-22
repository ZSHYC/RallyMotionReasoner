from __future__ import annotations

from typing import Any

RGR_CHECKPOINT_SCHEMA = "rallymotionreasoner.rgr.v1"


def validate_rgr_checkpoint(
    state: dict[str, Any],
    *,
    expected_graph_source: str | None = None,
    expected_event_backend: str | None = None,
) -> None:
    """Validate the structural contract needed to load a RGR checkpoint."""
    if not isinstance(state, dict) or state.get("schema") != RGR_CHECKPOINT_SCHEMA:
        schema = state.get("schema") if isinstance(state, dict) else None
        raise ValueError(f"unsupported RGR checkpoint schema: {schema}")
    required = {"model_state", "vocab", "label_maps", "config", "thresholds", "event_backend"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"RGR checkpoint is missing required fields: {missing}")
    if not isinstance(state["vocab"], dict) or not state["vocab"]:
        raise ValueError("RGR checkpoint has an empty vocabulary")
    label_maps = state["label_maps"]
    if not isinstance(label_maps, dict) or not label_maps:
        raise ValueError("RGR checkpoint has no label maps")
    for name, mapping in label_maps.items():
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"RGR label map is empty: {name}")
        ids = list(mapping.values())
        if any(not isinstance(value, int) for value in ids) or sorted(ids) != list(range(len(ids))):
            raise ValueError(f"RGR label map ids must be contiguous from zero: {name}")
    event_backend = str(state["event_backend"])
    if expected_event_backend is not None and event_backend != expected_event_backend:
        raise ValueError(f"RGR event backend mismatch: {event_backend} != {expected_event_backend}")
    if event_backend != "motion_region":
        raise ValueError(f"unsupported RGR event backend: {event_backend}")
    graph_source = str(state.get("graph_source") or state["config"].get("data", {}).get("graph_source") or "")
    if expected_graph_source is not None and graph_source != expected_graph_source:
        raise ValueError(f"RGR checkpoint graph source mismatch: {graph_source} != {expected_graph_source}")
    if int(state.get("visual_feature_dim", 0)) <= 0:
        raise ValueError("RGR checkpoint has an invalid visual feature dimension")
    threshold = (state.get("thresholds") or {}).get("evidence_threshold")
    if not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
        raise ValueError("RGR checkpoint requires an evidence threshold in [0, 1]")
