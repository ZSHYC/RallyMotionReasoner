from pathlib import Path

import numpy as np
import torch

from rallymotionreasoner.data.graph_qa import match_gold_frames_to_predicted_shots, tokenize
from rallymotionreasoner.evaluation.event_metrics import one_to_one_match
from rallymotionreasoner.event_detection.decoder import DecodedEvent
from rallymotionreasoner.event_detection.features import ball_frame_features
from rallymotionreasoner.event_detection.graph import build_predicted_graph
from rallymotionreasoner.event_detection.model import MotionRegionEventModel
from rallymotionreasoner.features.ball_trajectory import normalize_trajectory, trajectory_rows
from rallymotionreasoner.graph_reasoning.model import RallyGraphReasoner
from rallymotionreasoner.training.eval import prf
from rallymotionreasoner.training.losses import compute_loss


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


def test_rgr_padding_tokens_remain_finite() -> None:
    model = RallyGraphReasoner(
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


def test_rgr_consumes_structured_state_and_motion_contract() -> None:
    model = RallyGraphReasoner(
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
    assert output["ball_xy"].shape == (1, 2, 2)
    assert output["ball_visible_logits"].shape == (1, 2)
    assert output["contact_frame"].shape == (1, 2)
    assert torch.isfinite(output["evidence_logits"]).all()


def test_rgr_auxiliary_loss_and_multilabel_key_metric_are_finite() -> None:
    outputs = {
        "evidence_logits": torch.tensor([[0.0, -1.0]]),
        "key_action_logits": torch.tensor([[1.0, -1.0]]),
        "ball_xy": torch.full((1, 2, 2), 0.5),
        "ball_visible_logits": torch.zeros(1, 2),
        "contact_frame": torch.full((1, 2), 0.5),
        "level_1_logits": torch.zeros(1, 1),
    }
    batch = {
        "evidence": torch.tensor([[1.0, 0.0]]),
        "key": torch.tensor([[1.0, 1.0]]),
        "node_mask": torch.ones(1, 2, dtype=torch.bool),
        "ball_xy": torch.full((1, 2, 2), 0.5),
        "ball_mask": torch.ones(1, 2),
        "ball_visible": torch.ones(1, 2),
        "contact_frame": torch.full((1, 2), 0.5),
        "contact_mask": torch.ones(1, 2),
        "labels": {"level_1": torch.zeros(1, dtype=torch.long)},
    }
    loss = compute_loss(outputs, batch, {"ball_xy": 0.25, "ball_visible": 0.25, "contact_frame": 0.25})
    assert torch.isfinite(loss)
    assert prf({1, 2}, {1, 2}) == (1.0, 1.0, 1.0)


def test_region_motion_contract_and_bounce_graph_cue() -> None:
    rows = trajectory_rows(
        {"width": 100, "height": 50, "points": [
            {"frame": 0, "ball_x": 10, "ball_y": 20, "visible": True},
            {"frame": 1, "ball_x": 20, "ball_y": 20, "visible": True},
        ]},
        [0, 1],
        fps=25.0,
        width=100,
        height=50,
    )
    normalized = normalize_trajectory(
        rows,
        {"center": [0.0] * 8, "scale": [1.0] * 8, "clip": 5.0},
    )
    assert normalized.shape == (2, 10)
    hit = DecodedEvent(4, 0.16, 0.8, {}, {})
    bounce = DecodedEvent(5, 0.20, 0.7, {}, {}, event_type="bounce")
    graph = build_predicted_graph([hit, bounce], rally_id="r", frames_dir=Path("."), fps=25.0, num_frames=8)
    assert len(graph["strokes"]) == 1
    assert graph["bounce_events"][0]["frame"] == 5
    assert graph["strokes"][0]["bounce_after_gap"] == 1


def test_region_cross_modal_heads_have_expected_outputs() -> None:
    model = MotionRegionEventModel(attribute_maps={"hitter": {"near": 0, "far": 1}}).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 25, 11), torch.randn(1, 49, 3840))
    assert output["eventness_logit"].shape == (1,)
    assert output["type_logits"].shape == (1, 2)
    assert output["hitter_logits"].shape == (1, 2)
