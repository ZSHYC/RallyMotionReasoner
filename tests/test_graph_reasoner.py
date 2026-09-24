from pathlib import Path

import pytest
import torch

from rallymotionreasoner.graph_reasoning.graph_transformer import EDGE_TYPE_TO_ID
from rallymotionreasoner.graph_reasoning.model import RallyGraphReasoner
from rallymotionreasoner.graph_reasoning.motion_adapter import TennisMotionAdapter
from rallymotionreasoner.graph_reasoning.runtime import RGRCheckpointSelector


def test_motion_adapter_padding_does_not_change_valid_nodes() -> None:
    torch.manual_seed(0)
    adapter = TennisMotionAdapter(input_dim=8, hidden_dim=16, num_heads=4, dropout=0.0).eval()
    valid_features = torch.randn(1, 2, 8)
    padded_features = torch.cat([valid_features, torch.zeros(1, 1, 8)], dim=1)

    with torch.no_grad():
        output = adapter(valid_features, torch.ones(1, 2, dtype=torch.bool))
        padded_output = adapter(padded_features, torch.tensor([[True, True, False]]))

    assert torch.allclose(output, padded_output[:, :2])


def test_rgr_auxiliary_targets_do_not_change_model_outputs() -> None:
    torch.manual_seed(0)
    model = RallyGraphReasoner(
        vocab_size=16,
        label_maps={"level_1": {"A": 0}},
        visual_feature_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    batch = {
        "question_ids": torch.tensor([[2, 3]]),
        "node_ids": torch.tensor([[[4, 5], [6, 7]]]),
        "node_frames": torch.tensor([[0.2, 0.8]]),
        "visual_features": torch.randn(1, 2, 8),
        "ball_xy": torch.tensor([[[0.2, 0.3], [0.4, 0.5]]]),
        "ball_visible": torch.tensor([[1.0, 0.7]]),
        "ball_mask": torch.ones(1, 2),
        "contact_frame": torch.tensor([[0.2, 0.8]]),
        "contact_mask": torch.ones(1, 2),
        "node_mask": torch.ones(1, 2, dtype=torch.bool),
        "edge_index": torch.zeros(1, 1, 2, dtype=torch.long),
        "edge_type": torch.zeros(1, 1, dtype=torch.long),
        "edge_mask": torch.zeros(1, 1, dtype=torch.bool),
    }
    changed_targets = dict(batch)
    changed_targets.update(
        ball_xy=1.0 - batch["ball_xy"],
        ball_visible=1.0 - batch["ball_visible"],
        ball_mask=torch.zeros_like(batch["ball_mask"]),
        contact_frame=1.0 - batch["contact_frame"],
        contact_mask=torch.zeros_like(batch["contact_mask"]),
    )

    with torch.no_grad():
        output = model(batch)
        changed_output = model(changed_targets)

    assert output.keys() == changed_output.keys()
    assert all(torch.allclose(output[key], changed_output[key]) for key in output)


def test_rgr_forward_uses_paper_relations() -> None:
    assert set(EDGE_TYPE_TO_ID) == {"temporal_next", "same_player_next"}
    model = RallyGraphReasoner(
        vocab_size=32,
        label_maps={"level_1": {"A": 0, "B": 1}, "level_2": {"A1": 0}, "level_3": {"net_attack": 0}},
        visual_feature_dim=800,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    batch = {
        "question_ids": torch.tensor([[2, 3, 0, 0]]),
        "node_ids": torch.tensor([[[4, 5], [6, 7], [8, 9], [10, 11]]]),
        "node_frames": torch.tensor([[0.1, 0.3, 0.6, 1.0]]),
        "visual_features": torch.randn(1, 4, 800),
        "node_mask": torch.ones(1, 4, dtype=torch.bool),
        "edge_index": torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 2]]]),
        "edge_type": torch.tensor([[[0, 0, 0, 1]]]).reshape(1, 4),
        "edge_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    output = model(batch)
    assert output["evidence_logits"].shape == (1, 4)
    assert output["key_action_logits"].shape == (1, 4)
    assert output["level_1_logits"].shape == (1, 2)
    assert torch.allclose(output["evidence_weights"].sum(dim=-1), torch.ones(1), atol=1e-5)


def test_runtime_uses_checkpoint_training_stroke_limit() -> None:
    selector = object.__new__(RGRCheckpointSelector)
    selector.torch = torch
    selector.device = torch.device("cpu")
    selector.top_k = 8
    selector.evidence_threshold = 0.5
    selector.motion_feature_dim = 0
    selector.state = {
        "visual_feature_dim": 1,
        "vocab": {"<pad>": 0, "<unk>": 1},
        "label_maps": {},
        "config": {"data": {"max_strokes": 2, "use_edges": True}},
    }

    class Model:
        def __call__(self, batch):
            shape = batch["node_mask"].shape
            return {
                "evidence_logits": torch.ones(shape),
                "key_action_logits": torch.ones(shape),
            }

    selector.model = Model()
    graph = {
        "rally_id": "r",
        "strokes": [{"shot_id": idx + 1, "frame": idx, "parsed": {}} for idx in range(3)],
        "edges": [],
    }

    candidates = selector.select(graph, "why?", torch.zeros(3, 1), [0, 1, 2])

    assert [item["shot_id"] for item in candidates] == [1, 2]


@pytest.mark.parametrize(("checkpoint_top_k", "expected"), [(None, 8), (3, 3)])
def test_runtime_uses_checkpoint_top_k_with_legacy_default(
    tmp_path: Path, checkpoint_top_k: int | None, expected: int
) -> None:
    label_maps = {"level_1": {"null": 0}}
    model = RallyGraphReasoner(
        vocab_size=2,
        label_maps=label_maps,
        visual_feature_dim=1,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )
    state = {
        "schema": "rallymotionreasoner.rgr.v1",
        "model_state": model.state_dict(),
        "vocab": {"<pad>": 0, "<unk>": 1},
        "label_maps": label_maps,
        "config": {"data": {"graph_source": "motion_region", "include_label_tokens": False}},
        "thresholds": {"evidence_threshold": 0.5},
        "event_backend": "motion_region",
        "graph_source": "motion_region",
        "visual_feature_dim": 1,
        "hidden_dim": 4,
        "num_layers": 1,
        "num_heads": 1,
        "dropout": 0.0,
    }
    if checkpoint_top_k is not None:
        state["top_k"] = checkpoint_top_k
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(state, checkpoint)

    selector = RGRCheckpointSelector(checkpoint, device="cpu")

    assert selector.top_k == expected
    with pytest.raises(ValueError, match="top_k override"):
        RGRCheckpointSelector(checkpoint, device="cpu", top_k=expected + 1)
    if checkpoint_top_k == 3:
        state["top_k"] = 33
        torch.save(state, checkpoint)
        with pytest.raises(ValueError, match="RGR top_k must be between 1 and 32"):
            RGRCheckpointSelector(checkpoint, device="cpu")
