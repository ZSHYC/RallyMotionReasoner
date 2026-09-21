import numpy as np
import torch

from tennisvar.data.graph_qa import match_gold_frames_to_predicted_shots, tokenize
from tennisvar.event_parsing.features import ball_frame_features
from tennisvar.evaluation.event_metrics import one_to_one_match
from tennisvar.tactical_reasoning.model import TacticalGraphGuidedTemporalReasoner


def test_temporal_matching_maximizes_cardinality() -> None:
    assert one_to_one_match([12, 30], [0, 15], 16) == [(0, 0), (1, 1)]
    assert match_gold_frames_to_predicted_shots(
        [{"shot_id": 1, "frame": 12}, {"shot_id": 2, "frame": 30}], [0, 15], tolerance=16
    ) == {1, 2}


def test_question_tokenizer_keeps_cjk_coverage() -> None:
    assert tokenize("为什么近端球员选择上网？")


def test_invisible_ball_does_not_crash_or_create_motion() -> None:
    features, status = ball_frame_features(
        [0, 1], {"width": 100, "height": 100, "points": [{"frame": 0, "ball_x": None, "ball_y": None, "visible": False}]}
    )
    assert status == "EMPTY_OR_INVISIBLE"
    assert np.allclose(features, 0.0)


def test_tgtr_padding_tokens_remain_finite() -> None:
    model = TacticalGraphGuidedTemporalReasoner(
        vocab_size=16,
        label_maps={"level_1": {"A": 0}},
        visual_feature_dim=800,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    batch = {
        "question_ids": torch.tensor([[2, 3, 0]]),
        "node_ids": torch.tensor([[[4, 5], [0, 0]]]),
        "node_frames": torch.tensor([[0.1, 0.0]]),
        "visual_features": torch.randn(1, 2, 800),
        "node_mask": torch.tensor([[True, False]]),
        "edge_index": torch.zeros(1, 1, 2, dtype=torch.long),
        "edge_type": torch.zeros(1, 1, dtype=torch.long),
        "edge_mask": torch.zeros(1, 1, dtype=torch.bool),
    }
    with torch.no_grad():
        output = model(batch)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_tgtr_consumes_structured_state_and_motion_contract() -> None:
    model = TacticalGraphGuidedTemporalReasoner(
        vocab_size=16,
        label_maps={"level_1": {"A": 0}},
        visual_feature_dim=8,
        motion_feature_dim=3,
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
        "motion_features": torch.randn(1, 2, 3),
        "ball_xy": torch.tensor([[[0.2, 0.3], [0.4, 0.5]]]),
        "ball_visible": torch.tensor([[1.0, 0.7]]),
        "ball_mask": torch.ones(1, 2),
        "contact_frame": torch.tensor([[0.2, 0.8]]),
        "contact_mask": torch.ones(1, 2),
        "node_mask": torch.ones(1, 2, dtype=torch.bool),
        "edge_index": torch.tensor([[[0, 1]]]),
        "edge_type": torch.tensor([[0]]),
        "edge_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    with torch.no_grad():
        output = model(batch)
    assert output["evidence_logits"].shape == (1, 2)
    assert torch.isfinite(output["evidence_logits"]).all()
