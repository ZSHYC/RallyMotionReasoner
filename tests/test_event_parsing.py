from pathlib import Path

import torch

from tennisvar.data.graph_qa import load_motion_features, load_visual_features
from tennisvar.event_parsing.decoder import DecodedEvent
from tennisvar.event_parsing.features import EVENT_FEATURE_DIM, MOTION_DIM, VISUAL_DIM, shot_feature_payload
from tennisvar.event_parsing.graph import build_predicted_graph
from tennisvar.event_parsing.model import RegionFusionEventModel


def test_event_feature_dimensions_match_pipeline() -> None:
    assert (VISUAL_DIM, MOTION_DIM, EVENT_FEATURE_DIM) == (768, 24, 800)


def test_region_event_model_output_shapes() -> None:
    maps = {"hitter": {"near": 0, "far": 1}, "technique": {"forehand": 0, "backhand": 1}}
    model = RegionFusionEventModel(attribute_maps=maps, dropout=0.0).eval()
    output = model(torch.randn(2, 25, 11), torch.randn(2, 49, 3840))
    assert output["eventness_logit"].shape == (2,)
    assert output["type_logits"].shape == (2, 2)
    assert output["hitter_logits"].shape == (2, 2)


def test_hit_feature_export_matches_tgtr_loader(tmp_path: Path) -> None:
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
    model = RegionFusionEventModel(dropout=0).eval()
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
