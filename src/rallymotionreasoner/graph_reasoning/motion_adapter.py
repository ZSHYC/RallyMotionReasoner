from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None


if nn is not None:

    class TennisMotionAdapter(nn.Module):
        """Project fused 800-D event features into contextual 256-D stroke cues."""

        def __init__(self, input_dim: int = 800, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
            self.local = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.gate = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.output = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, features: Tensor, node_mask: Tensor | None = None) -> Tensor:
            base = self.input(features)
            local = self.local(base.transpose(1, 2)).transpose(1, 2)
            padding_mask = None if node_mask is None else ~node_mask.bool()
            global_context, _ = self.attention(
                base,
                base,
                base,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            context = self.dropout(torch.tanh(local + global_context))
            output = self.output(base + self.gate(torch.cat([base, context], dim=-1)) * context)
            return output if node_mask is None else output * node_mask.unsqueeze(-1).float()

else:

    class TennisMotionAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("TennisMotionAdapter requires torch")
