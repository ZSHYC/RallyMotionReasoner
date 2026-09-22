from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    Tensor = Any
    nn = None

from .evidence_router import EvidenceRouter
from .graph_transformer import TacticalGraphBatch, TacticalGraphTransformer
from .motion_adapter import TennisMotionAdapter
from .stroke_tokenizer import StrokeEventTokenizer

if nn is not None:

    class MultiScaleTemporalMixer(nn.Module):
        """Preserve contact-local detail while adding short-range context."""

        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.short = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
            self.medium = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim)
            self.mix = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
            self.gate = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.dropout = nn.Dropout(dropout)

        def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
            value = tokens if mask is None else tokens * mask.unsqueeze(-1).float()
            short = self.short(value.transpose(1, 2)).transpose(1, 2)
            medium = self.medium(value.transpose(1, 2)).transpose(1, 2)
            context = self.mix(torch.cat([short, medium], dim=-1))
            gate = self.gate(torch.cat([value, context], dim=-1))
            output = value + self.dropout(gate * context)
            return output if mask is None else output * mask.unsqueeze(-1).float()

    class RallyGraphReasoner(nn.Module):
        """Paper implementation of question-conditioned RGR."""

        def __init__(
            self,
            vocab_size: int,
            label_maps: dict[str, dict[str, int]],
            *,
            visual_feature_dim: int = 800,
            motion_feature_dim: int = 0,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.12,
        ) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.question_norm = nn.LayerNorm(hidden_dim)
            self.semantic_norm = nn.LayerNorm(hidden_dim)
            token_layer = nn.TransformerEncoderLayer(
                hidden_dim,
                max(1, min(num_heads, 4)),
                hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=1)
            self.visual_adapter = TennisMotionAdapter(
                input_dim=visual_feature_dim,
                hidden_dim=hidden_dim,
                num_heads=max(1, min(4, num_heads)),
                dropout=dropout,
            )
            self.motion_feature_dim = int(motion_feature_dim)
            self.state_adapter = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, hidden_dim),
                nn.GELU(),
            )
            self.motion_adapter = (
                nn.Sequential(nn.LayerNorm(self.motion_feature_dim), nn.Linear(self.motion_feature_dim, hidden_dim), nn.GELU())
                if self.motion_feature_dim > 0
                else None
            )
            self.local_temporal = MultiScaleTemporalMixer(hidden_dim, dropout)
            self.stroke_tokenizer = StrokeEventTokenizer(hidden_dim, dropout=dropout)
            self.graph = TacticalGraphTransformer(hidden_dim, num_layers, num_heads, dropout)
            self.evidence_router = EvidenceRouter(hidden_dim, dropout=dropout)
            self.ball_xy_head = nn.Linear(hidden_dim, 2)
            self.ball_visible_head = nn.Linear(hidden_dim, 1)
            self.contact_frame_head = nn.Linear(hidden_dim, 1)
            self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, len(mapping)) for name, mapping in label_maps.items()})

        def _sequence_embedding(self, token_ids: Tensor) -> Tensor:
            """Encode token order before pooling; padding never contributes to the pool."""
            shape = token_ids.shape
            flat = token_ids.reshape(-1, shape[-1])
            embeddings = self.token_embedding(flat)
            valid = flat.ne(0)
            safe_valid = valid.clone()
            empty = ~safe_valid.any(dim=1)
            safe_valid[empty, 0] = True
            position = torch.arange(shape[-1], device=flat.device).view(1, -1)
            div = torch.exp(
                torch.arange(0, embeddings.shape[-1], 2, device=flat.device, dtype=embeddings.dtype)
                * (-torch.log(torch.tensor(10000.0, device=flat.device, dtype=embeddings.dtype)) / embeddings.shape[-1])
            )
            positional = torch.zeros(1, shape[-1], embeddings.shape[-1], device=flat.device, dtype=embeddings.dtype)
            positional[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div)
            positional[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div[: positional[:, :, 1::2].shape[-1]])
            encoded = self.token_encoder(embeddings + positional, src_key_padding_mask=~safe_valid)
            weights = valid.float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return pooled.reshape(*shape[:-1], -1)

        def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
            question = self.question_norm(self._sequence_embedding(batch["question_ids"]))
            semantics = self.semantic_norm(self._sequence_embedding(batch["node_ids"]))
            visual = self.visual_adapter(batch["visual_features"], batch.get("node_mask"))
            state = torch.cat(
                [
                    batch.get("ball_xy", visual.new_zeros((*visual.shape[:2], 2))),
                    batch.get("ball_visible", visual.new_zeros(visual.shape[:2])).unsqueeze(-1),
                    batch.get("ball_mask", visual.new_zeros(visual.shape[:2])).unsqueeze(-1),
                    batch.get("contact_frame", visual.new_zeros(visual.shape[:2])).unsqueeze(-1),
                    batch.get("contact_mask", visual.new_zeros(visual.shape[:2])).unsqueeze(-1),
                ],
                dim=-1,
            )
            visual = visual + self.state_adapter(state)
            if self.motion_adapter is not None and batch.get("motion_features") is not None:
                motion = batch["motion_features"]
                if motion.shape[-1] != self.motion_feature_dim:
                    raise ValueError("motion feature dimension does not match RGR configuration")
                visual = visual + self.motion_adapter(motion)
            strokes = self.stroke_tokenizer(semantics, visual, batch["node_frames"])
            strokes = self.local_temporal(strokes, batch.get("node_mask"))
            graph = self.graph(
                TacticalGraphBatch(
                    node_tokens=strokes,
                    edge_index=batch["edge_index"],
                    edge_type=batch["edge_type"],
                    node_frames=batch.get("node_frames"),
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
                "ball_xy": torch.sigmoid(self.ball_xy_head(graph.node_tokens)),
                "ball_visible_logits": self.ball_visible_head(graph.node_tokens).squeeze(-1),
                "contact_frame": torch.sigmoid(self.contact_frame_head(graph.node_tokens)).squeeze(-1),
                **{f"{name}_logits": head(routed.routed_token) for name, head in self.heads.items()},
            }

else:

    class RallyGraphReasoner:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("RallyGraphReasoner requires torch")
