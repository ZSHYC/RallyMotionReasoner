from pathlib import Path

from tennisvar.config import load_paths
from tennisvar.configs import load_tgtr_vl_config


def test_paper_configuration_matches_feature_contract() -> None:
    config = load_tgtr_vl_config(Path("configs/tennisvar.yaml"))
    event = config["event_parsing"]
    assert event["appearance_dim"] + event["motion_dim"] + event["ball_dim"] == 800
    assert event["fused_dim"] == config["feature_extraction"]["feature_dim"] == 800
    assert config["data"]["graph_source"] == "f3ed"
    assert config["data"]["include_label_tokens"] is False
    assert config["training"]["epochs"] == 120
    assert event["epochs"] == 40
    assert event["batch_size"] == 64
    assert config["generation"]["epochs"] == 5
    assert config["generation"]["lora_rank"] == 32
    assert config["generation"]["effective_batch_size"] == 32


def test_default_paths_stay_inside_repository() -> None:
    paths = load_paths(Path("tests/does-not-exist.yaml"))
    root = Path.cwd().resolve()
    assert paths["project_root"] == root
    assert paths["artifacts_root"] == root / "artifacts"
