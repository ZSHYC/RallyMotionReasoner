from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tennisvar.schema_v2 import REQUIRED_SCHEMA_V2
from tennisvar.tracks import normalize_track

QWEN_ADAPTER_SCHEMA = "tennisvar.qwen_lora.v2"
MANIFEST_NAME = "tennisvar_manifest.json"


def load_qwen_adapter_manifest(adapter: Path, *, expected_track: str | None = None) -> dict[str, Any]:
    adapter = Path(adapter)
    path = adapter / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Qwen adapter manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != QWEN_ADAPTER_SCHEMA:
        raise ValueError(f"unsupported Qwen adapter manifest schema: {payload.get('schema')}")
    required = {"track", "dataset", "base_model", "output_schema_fields", "provenance_contract", "training_contract"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Qwen adapter manifest is missing fields: {missing}")
    if list(payload.get("output_schema_fields") or []) != list(REQUIRED_SCHEMA_V2):
        raise ValueError("Qwen adapter output schema fields do not match the runtime contract")
    if expected_track is not None and normalize_track(payload["track"]) != normalize_track(expected_track):
        raise ValueError(f"Qwen adapter track mismatch: {payload['track']} != {expected_track}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found: {adapter}")
    normalize_track(payload["track"])
    if payload.get("provenance_contract") != "predicted_event+tgtr_checkpoint":
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
