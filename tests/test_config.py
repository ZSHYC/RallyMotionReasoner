from pathlib import Path

import pytest

from rallymotionreasoner.config import load_paths
from rallymotionreasoner.configs import load_graph_reasoner_config, resume_config_matches, saved_time_bias_version


def test_paper_configuration_matches_feature_contract() -> None:
    config = load_graph_reasoner_config(Path("configs/rallymotionreasoner.yaml"))
    assert config["feature_extraction"]["feature_dim"] == 800
    assert config["data"]["graph_source"] == "motion_region"
    assert config["data"]["include_label_tokens"] is False
    assert config["training"]["epochs"] == 120
    assert config["model"]["time_bias_version"] == 2


def test_default_paths_stay_inside_repository() -> None:
    paths = load_paths(Path("tests/does-not-exist.yaml"))
    root = Path.cwd().resolve()
    assert paths["project_root"] == root
    assert paths["artifacts_root"] == root / "artifacts"


def test_legacy_trainer_config_can_resume_only_across_time_bucket_version(tmp_path: Path) -> None:
    current = load_graph_reasoner_config(Path("configs/rallymotionreasoner.yaml"))
    legacy = {**current, "model": {key: value for key, value in current["model"].items() if key != "time_bias_version"}}
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(
        Path("configs/rallymotionreasoner.yaml").read_text(encoding="utf-8").replace("  time_bias_version: 2\n", ""),
        encoding="utf-8",
    )

    assert load_graph_reasoner_config(legacy_path)["model"]["time_bias_version"] == 2
    assert resume_config_matches(legacy, current)
    without_top_k = {**legacy, "data": {key: value for key, value in legacy["data"].items() if key != "top_k"}}
    assert resume_config_matches(without_top_k, current)
    assert resume_config_matches(without_top_k, without_top_k)
    assert not resume_config_matches({**legacy, "model": {**legacy["model"], "hidden_dim": 128}}, current)
    assert saved_time_bias_version({"config": legacy}) == 1
    assert saved_time_bias_version({"config": current}) == 2


@pytest.mark.parametrize("top_k", [0, 33])
def test_graph_reasoner_rejects_candidates_exceeding_qwen_frame_budget(top_k: int) -> None:
    with pytest.raises(ValueError, match="data.top_k must be between 1 and 32"):
        load_graph_reasoner_config(Path("configs/rallymotionreasoner.yaml"), [f"data.top_k={top_k}"])
