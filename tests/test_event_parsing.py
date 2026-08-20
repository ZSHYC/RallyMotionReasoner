import torch

from tennisvar.event_parsing.features import BALL_FRAME_DIM, EVENT_FEATURE_DIM, MOTION_DIM, VISUAL_DIM
from tennisvar.event_parsing.model import EventModelConfig, EventParsingModule


def test_event_feature_dimensions_match_paper() -> None:
    assert (VISUAL_DIM, MOTION_DIM, BALL_FRAME_DIM, EVENT_FEATURE_DIM) == (768, 24, 8, 800)


def test_event_parser_output_shapes() -> None:
    maps = {"hitter": {"near": 0, "far": 1}, "technique": {"forehand": 0, "backhand": 1}}
    model = EventParsingModule(EventModelConfig(input_dim=800, hidden_dim=64, num_layers=2, num_heads=4), maps)
    output = model(torch.randn(2, 12, 800), torch.ones(2, 12, dtype=torch.bool))
    assert output["event_logits"].shape == (2, 12)
    assert output["frame_tokens"].shape == (2, 12, 64)
    assert output["hitter_logits"].shape == (2, 12, 2)
