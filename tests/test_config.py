from pathlib import Path

from tennisvar.config import load_paths
from tennisvar.configs import load_tgtr_vl_config


def test_paper_configuration_matches_feature_contract() -> None:
    config = load_tgtr_vl_config(Path("configs/tennisvar.yaml"))
    assert config["feature_extraction"]["feature_dim"] == 800
    assert config["data"]["graph_source"] == "region_fusion"
    assert config["data"]["include_label_tokens"] is False
    assert config["training"]["epochs"] == 120


def test_default_paths_stay_inside_repository() -> None:
    paths = load_paths(Path("tests/does-not-exist.yaml"))
    root = Path.cwd().resolve()
    assert paths["project_root"] == root
    assert paths["artifacts_root"] == root / "artifacts"
