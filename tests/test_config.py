from pathlib import Path

from rallymotionreasoner.config import load_paths
from rallymotionreasoner.configs import load_graph_reasoner_config


def test_paper_configuration_matches_feature_contract() -> None:
    config = load_graph_reasoner_config(Path("configs/rallymotionreasoner.yaml"))
    assert config["feature_extraction"]["feature_dim"] == 800
    assert config["data"]["graph_source"] == "motion_region"
    assert config["data"]["include_label_tokens"] is False
    assert config["training"]["epochs"] == 120


def test_default_paths_stay_inside_repository() -> None:
    paths = load_paths(Path("tests/does-not-exist.yaml"))
    root = Path.cwd().resolve()
    assert paths["project_root"] == root
    assert paths["artifacts_root"] == root / "artifacts"
