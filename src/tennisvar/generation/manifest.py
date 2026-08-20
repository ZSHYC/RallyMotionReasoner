from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tennisvar.schema_v2 import REQUIRED_SCHEMA_V2
from tennisvar.tracks import normalize_track

QWEN_ADAPTER_SCHEMA = "tennisvar.qwen_lora.v2"
MANIFEST_NAME = "tennisvar_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qwen_adapter_manifest(
    adapter: Path,
    *,
    expected_track: str | None = None,
    expected_base_model_config_sha256: str | None = None,
) -> dict[str, Any]:
    adapter = Path(adapter)
    path = adapter / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Qwen adapter manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != QWEN_ADAPTER_SCHEMA:
        raise ValueError(f"unsupported Qwen adapter manifest schema: {payload.get('schema')}")
    required = {
        "track", "dataset", "base_model", "base_model_config_sha256", "train_data_sha256", "val_data_sha256",
        "adapter_config_sha256", "provenance_contract", "training_contract",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Qwen adapter manifest is missing fields: {missing}")
    if list(payload.get("output_schema_fields") or []) != list(REQUIRED_SCHEMA_V2):
        raise ValueError("Qwen adapter output schema fields do not match the runtime contract")
    if expected_track is not None and normalize_track(payload["track"]) != normalize_track(expected_track):
        raise ValueError(f"Qwen adapter track mismatch: {payload['track']} != {expected_track}")
    if (
        expected_base_model_config_sha256 is not None
        and payload.get("base_model_config_sha256") != expected_base_model_config_sha256
    ):
        raise ValueError("Qwen adapter base-model config hash mismatch")
    adapter_config = adapter / "adapter_config.json"
    if not adapter_config.is_file() or file_sha256(adapter_config) != payload["adapter_config_sha256"]:
        raise ValueError("Qwen adapter_config.json hash mismatch")
    for field in ("base_model_config_sha256", "train_data_sha256", "val_data_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid Qwen adapter manifest hash: {field}")
    normalize_track(payload["track"])
    expected_provenance = "source_row+predicted_event+tgtr_checkpoint"
    if payload.get("provenance_contract") != expected_provenance:
        raise ValueError("Qwen adapter provenance contract does not match its training track")
    training_contract = payload.get("training_contract")
    required_training = {
        "epochs", "learning_rate", "gradient_accumulation_steps", "lora_rank", "lora_alpha",
        "lora_dropout", "save_steps",
    }
    if not isinstance(training_contract, dict) or required_training - set(training_contract):
        raise ValueError("Qwen adapter has an incomplete training contract")
    if (
        int(training_contract["epochs"]) <= 0
        or float(training_contract["learning_rate"]) <= 0
        or int(training_contract["gradient_accumulation_steps"]) <= 0
        or int(training_contract["lora_rank"]) <= 0
        or int(training_contract["lora_alpha"]) <= 0
        or not 0.0 <= float(training_contract["lora_dropout"]) < 1.0
        or int(training_contract["save_steps"]) <= 0
    ):
        raise ValueError("Qwen adapter training contract has invalid values")
    return payload
