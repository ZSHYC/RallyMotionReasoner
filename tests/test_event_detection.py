from pathlib import Path

import torch

from rallymotionreasoner.data.graph_qa import load_motion_features, load_visual_features
from rallymotionreasoner.event_detection.decoder import DecodedEvent
from rallymotionreasoner.event_detection.features import EVENT_FEATURE_DIM, MOTION_DIM, VISUAL_DIM, shot_feature_payload
from rallymotionreasoner.event_detection.graph import build_predicted_graph
from rallymotionreasoner.event_detection.model import MotionRegionEventModel, TrajectoryExpert, VisualExpert
from rallymotionreasoner.event_detection.runtime import _decode


def test_event_feature_dimensions_match_pipeline() -> None:
    assert (VISUAL_DIM, MOTION_DIM, EVENT_FEATURE_DIM) == (768, 24, 800)


def test_region_event_model_output_shapes() -> None:
    maps = {"hitter": {"near": 0, "far": 1}, "technique": {"forehand": 0, "backhand": 1}}
    model = MotionRegionEventModel(attribute_maps=maps, dropout=0.0).eval()
    output = model(torch.randn(2, 25, 11), torch.randn(2, 49, 3840))
    assert output["eventness_logit"].shape == (2,)
    assert output["type_logits"].shape == (2, 2)
    assert output["hitter_logits"].shape == (2, 2)


def test_hit_feature_export_matches_rgr_loader(tmp_path: Path) -> None:
    graph = build_predicted_graph(
        [DecodedEvent(2, 0.08, 0.8, {}, {}), DecodedEvent(1, 0.04, 0.9, {}, {}, event_type="bounce")],
        rally_id="r", frames_dir=tmp_path, fps=25, num_frames=3,
    )
    features = torch.arange(2400).reshape(3, 800).float()
    payload = shot_feature_payload(graph, [0, 1, 2], features, split="train")
    (tmp_path / "train").mkdir()
    torch.save(payload, tmp_path / "train/r.pt")
    assert load_visual_features(tmp_path, "train", "r", [1], 800, strict=True) == features[2:3].tolist()
    assert load_motion_features(tmp_path, "train", "r", [1], 24) == features[2:3, 768:792].tolist()
    assert payload["shot_frames"] == [2]


def test_fusion_ignores_values_in_masked_slots() -> None:
    model = MotionRegionEventModel(dropout=0).eval()
    trajectory = torch.randn(1, 5, 11)
    trajectory[..., 9] = 0
    visual = torch.randn(1, 5, 3840)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    with torch.no_grad():
        first = model(trajectory, visual, mask)
        trajectory[..., :9] *= 100
        second = model(trajectory, visual * 100, mask)
    for name in first:
        assert torch.isfinite(first[name]).all()
        assert torch.equal(first[name], second[name])


def test_experts_exclude_edge_padding_from_temporal_features() -> None:
    cases = [
        (TrajectoryExpert().eval(), torch.randn(1, 3, 11)),
        (VisualExpert().eval(), torch.randn(1, 3, 3840)),
    ]
    cases[0][1][..., 9] = 1
    padding_mask = torch.tensor([[False, False, True, True, True]])
    for model, valid_values in cases:
        padded = torch.zeros(1, 5, valid_values.shape[-1])
        padded[:, 2:] = valid_values
        with torch.no_grad():
            valid_tokens, _ = model.encode(valid_values)
            padded_tokens, padded_pool = model.encode(padded, padding_mask=padding_mask)
        expected_pool = torch.cat(
            (valid_tokens[:, 0], valid_tokens.mean(dim=1), valid_tokens.amax(dim=1)), dim=-1
        )
        torch.testing.assert_close(padded_tokens[:, 2:], valid_tokens)
        assert torch.count_nonzero(padded_tokens[:, :2]) == 0
        torch.testing.assert_close(padded_pool, expected_pool)


def test_experts_batch_complete_rows_when_edge_padding_is_mixed() -> None:
    masks = torch.tensor(
        [
            [False, False, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    cases = [
        (TrajectoryExpert().eval(), torch.randn(4, 5, 11)),
        (VisualExpert().eval(), torch.randn(4, 5, 3840)),
    ]
    cases[0][1][..., 9] = 1
    for model, values in cases:
        with torch.no_grad():
            expected = [
                model.encode(values[index : index + 1], padding_mask=masks[index : index + 1])
                for index in range(len(values))
            ]
        batch_sizes = []
        handle = model.gru.register_forward_pre_hook(
            lambda _module, inputs: batch_sizes.append(inputs[0].shape[0])
        )
        with torch.no_grad():
            tokens, pooled = model.encode(values, padding_mask=masks)
        handle.remove()
        assert batch_sizes == [2, 1, 1]
        for index, (expected_tokens, expected_pool) in enumerate(expected):
            torch.testing.assert_close(tokens[index], expected_tokens[0])
            torch.testing.assert_close(pooled[index], expected_pool[0])


def test_decode_uses_one_fps_scaled_suppression_window_for_both_types() -> None:
    scores = torch.tensor([[0.90, 0.85], [0.10, 0.80], [0.10, 0.70]])
    events = _decode(scores, [100, 109, 111], fps=50.0, threshold=0.4, radius=5)
    assert [(event.frame, event.event_type) for event in events] == [(100, "hit"), (111, "bounce")]
