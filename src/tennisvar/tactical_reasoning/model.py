from __future__ import annotations

from typing import Any

try:
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    Tensor = Any
    nn = None

from .evidence_router import EvidenceRouter
from .graph_transformer import TacticalGraphBatch, TacticalGraphTransformer
from .motion_adapter import TennisMotionAdapter
from .stroke_tokenizer import StrokeEventTokenizer

if nn is not None:

    class TacticalGraphGuidedTemporalReasoner(nn.Module):
        """Paper implementation of question-conditioned TGTR."""

        def __init__(
            self,
            vocab_size: int,
            label_maps: dict[str, dict[str, int]],
            *,
            visual_feature_dim: int = 800,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.12,
        ) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.question_norm = nn.LayerNorm(hidden_dim)
            self.semantic_norm = nn.LayerNorm(hidden_dim)
            self.visual_adapter = TennisMotionAdapter(
                input_dim=visual_feature_dim,
                hidden_dim=hidden_dim,
                num_heads=max(1, min(4, num_heads)),
                dropout=dropout,
            )
            self.stroke_tokenizer = StrokeEventTokenizer(hidden_dim, dropout=dropout)
            self.graph = TacticalGraphTransformer(hidden_dim, num_layers, num_heads, dropout)
            self.evidence_router = EvidenceRouter(hidden_dim, dropout=dropout)
            self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, len(mapping)) for name, mapping in label_maps.items()})

        def _mean_embedding(self, token_ids: Tensor) -> Tensor:
            embeddings = self.token_embedding(token_ids)
            mask = token_ids.ne(0).float().unsqueeze(-1)
            return (embeddings * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)

        def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
            question = self.question_norm(self._mean_embedding(batch["question_ids"]))
            semantics = self.semantic_norm(self._mean_embedding(batch["node_ids"]))
            visual = self.visual_adapter(batch["visual_features"], batch.get("node_mask"))
            strokes = self.stroke_tokenizer(semantics, visual, batch["node_frames"])
            graph = self.graph(
                TacticalGraphBatch(
                    node_tokens=strokes,
                    edge_index=batch["edge_index"],
                    edge_type=batch["edge_type"],
                    node_mask=batch.get("node_mask"),
                    edge_mask=batch.get("edge_mask"),
                )
            )
            routed = self.evidence_router(
                graph.node_tokens,
                graph.graph_token,
                question,
                batch.get("node_mask"),
            )
            return {
                "evidence_logits": routed.evidence_logits,
                "key_action_logits": routed.key_action_logits,
                "evidence_weights": routed.evidence_weights,
                "graph_token": routed.routed_token,
                **{f"{name}_logits": head(routed.routed_token) for name, head in self.heads.items()},
            }

else:

    class TacticalGraphGuidedTemporalReasoner:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("TacticalGraphGuidedTemporalReasoner requires torch")
