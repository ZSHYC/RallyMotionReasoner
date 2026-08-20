from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .model import EventModelConfig, EventParsingModule

CHECKPOINT_SCHEMA = "tennisvar.f3ed.v1"


def data_fingerprint(source_hashes: dict[str, str], feature_hashes: dict[str, str]) -> str:
    payload = json.dumps(
        {"source_hashes": source_hashes, "feature_hashes": feature_hashes},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_event_checkpoint_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        schema = payload.get("schema") if isinstance(payload, dict) else None
        raise ValueError(f"unsupported event checkpoint schema: {schema}")
    required = {
        "model_config", "attribute_maps", "source_hashes", "feature_hashes", "data_fingerprint", "decoder", "run_manifest", "state_dict"
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"event checkpoint is missing required fields: {missing}")
    source_hashes = payload["source_hashes"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != {"train", "val", "test"}:
        raise ValueError("event checkpoint source hashes must contain exactly train, val and test")
    for split, digest in source_hashes.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ValueError(f"invalid event checkpoint source hash: {split}")
    feature_hashes = payload["feature_hashes"]
    if not isinstance(feature_hashes, dict) or set(feature_hashes) != {"train", "val", "test"}:
        raise ValueError("event checkpoint feature hashes must contain exactly train, val and test")
    for split, digest in feature_hashes.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ValueError(f"invalid event checkpoint feature hash: {split}")
    if payload.get("data_fingerprint") != data_fingerprint(source_hashes, feature_hashes):
        raise ValueError("event checkpoint data fingerprint is internally inconsistent")
    attribute_maps = payload["attribute_maps"]
    if not isinstance(attribute_maps, dict) or not attribute_maps:
        raise ValueError("event checkpoint has no attribute maps")
    for field, mapping in attribute_maps.items():
        if not isinstance(mapping, dict) or not mapping or len(set(mapping.values())) != len(mapping):
            raise ValueError(f"invalid event attribute map: {field}")
        if any(not isinstance(index, int) for index in mapping.values()):
            raise ValueError(f"non-integer event attribute map id: {field}")
    model_config = payload["model_config"]
    if not isinstance(model_config, dict) or int(model_config.get("input_dim") or 0) <= 0:
        raise ValueError("event checkpoint has an invalid model config")
    decoder = payload["decoder"]
    if not isinstance(decoder, dict) or not 0.0 <= float(decoder.get("threshold", -1.0)) <= 1.0:
        raise ValueError("event checkpoint has an invalid decoder")
    if int(decoder.get("min_separation_frames") or 0) <= 0 or int(decoder.get("top_k") or 0) <= 0:
        raise ValueError("event checkpoint decoder separation/top_k must be positive")
    manifest = payload["run_manifest"]
    if not isinstance(manifest, dict) or manifest.get("schema") != "tennisvar.run.v1":
        raise ValueError("event checkpoint has an invalid run manifest")
    if manifest.get("source_hashes") != source_hashes:
        raise ValueError("event checkpoint run manifest source hashes do not match")
    if manifest.get("feature_hashes") != feature_hashes:
        raise ValueError("event checkpoint run manifest feature hashes do not match")
    training_config = manifest.get("config")
    if not isinstance(training_config, dict):
        raise ValueError("event checkpoint run manifest has no training config")
    expected_config_hash = hashlib.sha256(json.dumps(training_config, sort_keys=True).encode()).hexdigest()
    if manifest.get("config_sha256") != expected_config_hash:
        raise ValueError("event checkpoint run manifest config hash mismatch")
    for field in ("hidden_dim", "num_layers", "num_heads", "dropout"):
        if field in training_config and model_config.get(field) != training_config.get(field):
            raise ValueError(f"event checkpoint model/training config mismatch: {field}")
    backend = str(manifest.get("event_feature_backend") or "")
    feature_contract = manifest.get("feature_contract")
    if (
        not isinstance(feature_contract, dict)
        or feature_contract.get("backend") != backend
        or int(feature_contract.get("feature_dim") or 0) != int(model_config.get("input_dim") or 0)
        or not isinstance(feature_contract.get("weights_sha256"), str)
        or len(feature_contract["weights_sha256"]) != 64
        or any(ch not in "0123456789abcdef" for ch in feature_contract["weights_sha256"].lower())
    ):
        raise ValueError("event checkpoint has an invalid feature contract")
    tracknet_hash = manifest.get("tracknet_checkpoint_sha256")
    if "tracknet8_real" in backend and (
        not isinstance(tracknet_hash, str)
        or len(tracknet_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in tracknet_hash.lower())
        or feature_contract.get("tracknet_checkpoint_sha256") != tracknet_hash
    ):
        raise ValueError("real-TrackNet event checkpoint has no valid TrackNet checkpoint hash")


def save_event_checkpoint(
    path: Path,
    model: Any,
    *,
    attribute_maps: dict[str, dict[str, int]],
    source_hashes: dict[str, str],
    feature_hashes: dict[str, str],
    decoder: dict[str, Any],
    metrics: dict[str, Any],
    run_manifest: dict[str, Any],
) -> None:
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model.config.to_json(),
            "attribute_maps": attribute_maps,
            "source_hashes": source_hashes,
            "feature_hashes": feature_hashes,
            "data_fingerprint": data_fingerprint(source_hashes, feature_hashes),
            "decoder": decoder,
            "metrics": metrics,
            "run_manifest": run_manifest,
            "state_dict": model.state_dict(),
        }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_event_checkpoint(path: Path, *, expected_data_fingerprint: str | None = None, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    validate_event_checkpoint_payload(payload)
    attribute_maps = payload["attribute_maps"]
    if expected_data_fingerprint and payload.get("data_fingerprint") != expected_data_fingerprint:
        raise ValueError("event checkpoint data fingerprint mismatch")
    config = EventModelConfig(**payload["model_config"])
    model = EventParsingModule(config, attribute_maps)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
