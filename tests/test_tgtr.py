import torch

from tennisvar.tactical_reasoning.graph_transformer import EDGE_TYPE_TO_ID
from tennisvar.tactical_reasoning.model import TacticalGraphGuidedTemporalReasoner


def test_tgtr_forward_uses_paper_relations() -> None:
    assert set(EDGE_TYPE_TO_ID) == {"temporal_next", "same_player_next"}
    model = TacticalGraphGuidedTemporalReasoner(
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
