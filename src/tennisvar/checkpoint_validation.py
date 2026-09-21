from __future__ import annotations

import hashlib
import json
from typing import Any

TGTR_CHECKPOINT_SCHEMA = "tennisvar.tgtr.v1"


def _json_hash(value: Any, *, compact: bool = False) -> str:
    kwargs: dict[str, Any] = {"sort_keys": True}
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **kwargs).encode("utf-8")).hexdigest()


def validate_tgtr_checkpoint(
    state: dict[str, Any],
    *,
    expected_graph_source: str | None = None,
    expected_event_backend: str | None = None,
) -> None:
    """Reject structurally incompatible or internally inconsistent TGTR checkpoints."""
    if not isinstance(state, dict) or state.get("schema") != TGTR_CHECKPOINT_SCHEMA:
        schema = state.get("schema") if isinstance(state, dict) else None
        raise ValueError(f"unsupported TGTR checkpoint schema: {schema}")
    required = {"model_state", "vocab", "label_maps", "config", "data_hashes", "data_fingerprint"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"TGTR checkpoint is missing required fields: {missing}")
    if not isinstance(state["vocab"], dict) or not state["vocab"]:
        raise ValueError("TGTR checkpoint has an empty vocabulary")
    label_maps = state["label_maps"]
    if not isinstance(label_maps, dict) or not label_maps:
        raise ValueError("TGTR checkpoint has no label maps")
    for name, mapping in label_maps.items():
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"TGTR label map is empty: {name}")
        ids = list(mapping.values())
        if any(not isinstance(value, int) for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"TGTR label map has invalid or duplicate ids: {name}")
    data_hashes = state["data_hashes"]
    if not isinstance(data_hashes, dict) or not data_hashes:
        raise ValueError("TGTR checkpoint has no data hashes")
    required_data_hashes = {
        "train_refs", "val_refs", "train_graphs", "val_graphs",
        "train_feature_export", "val_feature_export", "test_feature_export",
        "train_features", "val_features", "test_features",
    }
    missing_data_hashes = sorted(required_data_hashes - set(data_hashes))
    if missing_data_hashes:
        raise ValueError(f"TGTR checkpoint is missing immutable data hashes: {missing_data_hashes}")
    for name, digest in data_hashes.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ValueError(f"invalid TGTR data hash for {name}")
    expected_fingerprint = _json_hash(data_hashes)
    if state["data_fingerprint"] != expected_fingerprint:
        raise ValueError("TGTR checkpoint data fingerprint is internally inconsistent")
    config_hash = _json_hash(state["config"], compact=True)
    if state.get("config_hash") != config_hash:
        raise ValueError("TGTR checkpoint config hash is internally inconsistent")
    run_manifest = state.get("run_manifest") or {}
    if run_manifest.get("config_hash") != config_hash or run_manifest.get("data_fingerprint") != expected_fingerprint:
        raise ValueError("TGTR checkpoint run manifest does not match checkpoint hashes")
    event_backend = str(
        state.get("event_backend")
        or run_manifest.get("event_backend")
        or state["config"].get("data", {}).get("event_backend")
        or "f3ed"
    )
    if expected_event_backend is not None and event_backend != expected_event_backend:
        raise ValueError(f"TGTR event backend mismatch: {event_backend} != {expected_event_backend}")
    if event_backend == "f3ed":
        for field in ("upstream_event_checkpoint_sha256", "upstream_event_data_fingerprint"):
            digest = state.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest.lower())
                or run_manifest.get(field) != digest
            ):
                raise ValueError(f"TGTR checkpoint has invalid upstream event lineage: {field}")
    elif event_backend != "region_fusion":
        raise ValueError(f"unsupported TGTR event backend: {event_backend}")
    graph_source = str(state.get("graph_source") or state["config"].get("data", {}).get("graph_source") or "")
    if expected_graph_source is not None and graph_source != expected_graph_source:
        raise ValueError(f"TGTR checkpoint graph source mismatch: {graph_source} != {expected_graph_source}")
    if int(state.get("visual_feature_dim", 0)) <= 0:
        raise ValueError("TGTR checkpoint has an invalid visual feature dimension")
