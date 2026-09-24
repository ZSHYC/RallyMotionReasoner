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
    node_frames: Tensor | None = None
    node_mask: Tensor | None = None
    edge_mask: Tensor | None = None


@dataclass
class TacticalGraphOutput:
    node_tokens: Tensor
    graph_token: Tensor


if nn is not None:

    class RelationTemporalBlock(nn.Module):
        """A graph block with directed relation messages and time biases."""

        def __init__(self, hidden_dim: int, num_heads: int, num_relations: int, dropout: float) -> None:
            super().__init__()
            self.edge_embedding = nn.Embedding(num_relations, hidden_dim)
            self.message = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.edge_bias = nn.Embedding(num_relations, num_heads)
            self.time_bias = nn.Embedding(17, num_heads)
            self.zero_time_bucket = True
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )

        def _messages(self, batch: TacticalGraphBatch) -> tuple[Tensor, Tensor]:
            tokens = batch.node_tokens
            output = torch.zeros_like(tokens)
            bias = torch.zeros(
                tokens.size(0), self.edge_bias.embedding_dim, tokens.size(1), tokens.size(1),
                device=tokens.device, dtype=tokens.dtype,
            )
            edge_index, edge_type = batch.edge_index, batch.edge_type
            if edge_index.ndim == 2:
                edge_index = edge_index.unsqueeze(0).expand(tokens.size(0), -1, -1)
                edge_type = edge_type.unsqueeze(0).expand(tokens.size(0), -1)
            edge_mask = batch.edge_mask
            if edge_mask is None:
                edge_mask = torch.ones(edge_type.shape, dtype=torch.bool, device=edge_type.device)
            for batch_index in range(tokens.size(0)):
                valid = edge_mask[batch_index].bool()
                if not valid.any():
                    continue
                source = edge_index[batch_index, valid, 0].long()
                target = edge_index[batch_index, valid, 1].long()
                if source.min() < 0 or target.min() < 0:
                    raise ValueError("graph edge indices must be non-negative")
                if source.max() >= tokens.size(1) or target.max() >= tokens.size(1):
                    raise ValueError("graph edge index exceeds the number of stroke nodes")
                relation_ids = edge_type[batch_index, valid].long()
                relation = self.edge_embedding(relation_ids)
                source_tokens = tokens[batch_index].index_select(0, source)
                messages = self.message(torch.cat([source_tokens, relation], dim=-1))
                degree = torch.zeros(tokens.size(1), device=tokens.device, dtype=messages.dtype)
                degree.index_add_(0, target, torch.ones_like(target, dtype=messages.dtype))
                messages = messages / degree.index_select(0, target).clamp_min(1.0).sqrt().unsqueeze(-1)
                output[batch_index].index_add_(0, target, messages)
                bias[batch_index, :, target, source] = self.edge_bias(relation_ids).transpose(0, 1)
            return output, bias

        def _attention_bias(self, batch: TacticalGraphBatch, relation_bias: Tensor) -> Tensor:
            batch_size, nodes = batch.node_tokens.shape[:2]
            frames = batch.node_frames
            if frames is None:
                frames = torch.arange(nodes, device=batch.node_tokens.device, dtype=batch.node_tokens.dtype)
                frames = frames.unsqueeze(0).expand(batch_size, -1)
            delta = frames.unsqueeze(2) - frames.unsqueeze(1)
            gap = (delta.abs() * 8.0).long().clamp(max=7)
            signed = gap + 1
            signed = torch.where(delta < 0, -signed, signed)
            if self.zero_time_bucket:
                signed = torch.where(delta == 0, 0, signed)
            time_index = (signed + 8).clamp(0, 16)
            time_bias = self.time_bias(time_index).permute(0, 3, 1, 2)
            return (relation_bias + time_bias).reshape(batch_size * self.attention.num_heads, nodes, nodes)

        def forward(self, batch: TacticalGraphBatch) -> Tensor:
            messages, relation_bias = self._messages(batch)
            tokens = batch.node_tokens + messages
            normalized = self.norm1(tokens)
            padding_mask = None
            if batch.node_mask is not None:
                padding_mask = (~batch.node_mask.bool()).to(tokens.dtype) * -1e4
            attention, _ = self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=padding_mask,
                attn_mask=self._attention_bias(batch, relation_bias),
                need_weights=False,
            )
            tokens = tokens + attention
            tokens = tokens + self.ffn(self.norm2(tokens))
            if batch.node_mask is not None:
                tokens = tokens * batch.node_mask.unsqueeze(-1).float()
            return tokens


    class TacticalGraphTransformer(nn.Module):
        """Multi-layer relation- and time-biased graph Transformer over strokes."""

        def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1) -> None:
            super().__init__()
            if hidden_dim % num_heads:
                raise ValueError("hidden_dim must be divisible by num_heads")
            self.layers = nn.ModuleList(
                RelationTemporalBlock(hidden_dim, num_heads, len(EDGE_TYPES), dropout)
                for _ in range(max(1, num_layers))
            )
            self.pool_score = nn.Linear(hidden_dim, 1)
            self.pool = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.time_bias_version = 2

        def set_time_bias_version(self, version: int) -> None:
            if version not in {1, 2}:
                raise ValueError(f"unsupported RGR time bias version: {version}")
            self.time_bias_version = version
            for layer in self.layers:
                layer.zero_time_bucket = version >= 2

        def forward(self, batch: TacticalGraphBatch) -> TacticalGraphOutput:
            tokens = batch.node_tokens
            for layer in self.layers:
                tokens = layer(
                    TacticalGraphBatch(
                        tokens,
                        batch.edge_index,
                        batch.edge_type,
                        batch.node_frames,
                        batch.node_mask,
                        batch.edge_mask,
                    )
                )
            scores = self.pool_score(tokens).squeeze(-1)
            if batch.node_mask is not None:
                scores = scores.masked_fill(~batch.node_mask.bool(), -1e4)
            weights = torch.softmax(scores, dim=-1)
            if batch.node_mask is not None:
                weights = weights * batch.node_mask.float()
            pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
            return TacticalGraphOutput(tokens, self.pool(pooled))

else:

    class TacticalGraphTransformer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("TacticalGraphTransformer requires torch")
