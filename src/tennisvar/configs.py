from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import DEFAULT_CONFIG, load_paths

PAPER_FEATURE_DIM = 800


def _apply_override(config: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must use key=value: {item}")
    dotted, raw_value = item.split("=", 1)
    target = config
    parts = dotted.strip().split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override nested key below {part}")
        target = child
    target[parts[-1]] = yaml.safe_load(raw_value)


def validate_paper_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    features = config.get("feature_extraction", {})
    flags = config.get("module_flags", {})
    errors = []
    if str(data.get("graph_source")) != "f3ed":
        errors.append("data.graph_source must be f3ed (predicted events)")
    if bool(data.get("include_label_tokens", True)):
        errors.append("data.include_label_tokens must be false")
    if not bool(data.get("use_edges", False)):
        errors.append("data.use_edges must be true")
    if int(features.get("feature_dim", 0)) != PAPER_FEATURE_DIM:
        errors.append(f"feature_extraction.feature_dim must be {PAPER_FEATURE_DIM}")
    if str(flags.get("reasoner")) != "relation_temporal_graph_chain":
        errors.append("module_flags.reasoner must be relation_temporal_graph_chain")
    if str(flags.get("vlm_adapter")) != "structured_prompt":
        errors.append("module_flags.vlm_adapter must be structured_prompt")
    if errors:
        raise ValueError("invalid TennisVAR paper configuration: " + "; ".join(errors))


def load_tgtr_vl_config(experiment_config: Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(experiment_config or "configs/tennisvar.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"experiment configuration must be a mapping: {path}")
    for item in overrides or []:
        _apply_override(config, item)
    validate_paper_config(config)
    return config


def resolve_paths_config(paths_config: Path | None = None) -> dict[str, Path]:
    return load_paths(paths_config or DEFAULT_CONFIG)


def tgtr_vl_feature_root(paths: dict[str, Path], config: dict[str, Any]) -> Path:
    features = config["feature_extraction"]
    return paths["artifacts_root"] / str(features["feature_root_name"]) / str(features["feature_set"])


def dump_simple_yaml(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=False)


def module_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": config.get("architecture"),
        "event_space": config.get("data", {}).get("graph_source"),
        "event_feature_dim": config.get("feature_extraction", {}).get("feature_dim"),
        "hidden_dim": config.get("model", {}).get("hidden_dim"),
        "graph_relations": ["temporal_next", "same_player_next"],
        "graph_attention": "relation_bias+signed_time_bias",
        "temporal_mixer": "depthwise_k3_k5_gated",
        "structured_state_dim": 6,
        "reasoner": config.get("module_flags", {}).get("reasoner"),
        "generator_interface": config.get("module_flags", {}).get("vlm_adapter"),
        "loss_weights": config.get("loss_weights", {}),
    }
