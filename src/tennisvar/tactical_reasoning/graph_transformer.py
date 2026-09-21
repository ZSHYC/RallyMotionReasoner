from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None


EDGE_TYPES = ["temporal_next", "same_player_next"]
EDGE_TYPE_TO_ID = {name: index for index, name in enumerate(EDGE_TYPES)}


@dataclass
class TacticalGraphBatch:
    node_tokens: Tensor
    edge_index: Tensor
    edge_type: Tensor
    node_mask: Tensor | None = None
    edge_mask: Tensor | None = None


@dataclass
class TacticalGraphOutput:
    node_tokens: Tensor
    graph_token: Tensor


if nn is not None:

    class TacticalGraphTransformer(nn.Module):
        """Relation-aware encoder over temporal and same-player stroke edges."""

        def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1) -> None:
            super().__init__()
            self.edge_embedding = nn.Embedding(len(EDGE_TYPES), hidden_dim)
            self.message = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.pool = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        def _messages(self, batch: TacticalGraphBatch) -> Tensor:
            output = torch.zeros_like(batch.node_tokens)
            if batch.edge_index.numel() == 0:
                return output
            edge_index = batch.edge_index
            edge_type = batch.edge_type
            if edge_index.ndim == 2:
                edge_index = edge_index.unsqueeze(0).expand(batch.node_tokens.size(0), -1, -1)
                edge_type = edge_type.unsqueeze(0).expand(batch.node_tokens.size(0), -1)
            edge_mask = batch.edge_mask
            if edge_mask is None:
                edge_mask = torch.ones(edge_type.shape, dtype=torch.bool, device=edge_type.device)
            for batch_index in range(batch.node_tokens.size(0)):
                valid = edge_mask[batch_index].bool()
                if not valid.any():
                    continue
                source = edge_index[batch_index, valid, 0].long()
                target = edge_index[batch_index, valid, 1].long()
                if source.min() < 0 or target.min() < 0:
                    raise ValueError("graph edge indices must be non-negative")
                if source.max() >= batch.node_tokens.size(1) or target.max() >= batch.node_tokens.size(1):
                    raise ValueError("graph edge index exceeds the number of stroke nodes")
                relation = self.edge_embedding(edge_type[batch_index, valid].long())
                source_tokens = batch.node_tokens[batch_index].index_select(0, source)
                messages = self.message(torch.cat([source_tokens, relation], dim=-1))
                # Degree normalization prevents a node with several incoming
                # relations from dominating solely because it has more edges.
                degree = torch.zeros(batch.node_tokens.size(1), device=target.device, dtype=messages.dtype)
                degree.index_add_(0, target, torch.ones_like(target, dtype=messages.dtype))
                messages = messages / degree.index_select(0, target).clamp_min(1.0).sqrt().unsqueeze(-1)
                output[batch_index].index_add_(0, target, messages)
            return output

        def forward(self, batch: TacticalGraphBatch) -> TacticalGraphOutput:
            tokens = batch.node_tokens + self._messages(batch)
            padding_mask = None if batch.node_mask is None else ~batch.node_mask.bool()
            encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
            if batch.node_mask is None:
                pooled = encoded.mean(dim=1)
            else:
                mask = batch.node_mask.float().unsqueeze(-1)
                pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return TacticalGraphOutput(encoded, self.pool(pooled))

else:

    class TacticalGraphTransformer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("TacticalGraphTransformer requires torch")
