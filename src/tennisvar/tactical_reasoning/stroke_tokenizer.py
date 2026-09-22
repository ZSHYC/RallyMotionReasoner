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

    class StrokeEventTokenizer(nn.Module):
        """Fuse semantic attributes, event visual cues, and normalized contact time."""

        def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.time_projection = nn.Linear(1, hidden_dim)
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output = nn.LayerNorm(hidden_dim)

        def forward(self, semantic_tokens: Tensor, visual_tokens: Tensor, contact_times: Tensor) -> Tensor:
            time_tokens = self.time_projection(contact_times.unsqueeze(-1))
            joint = self.fusion(torch.cat([semantic_tokens, visual_tokens, semantic_tokens * visual_tokens], dim=-1))
            return self.output(semantic_tokens + visual_tokens + time_tokens + joint)

else:

    class StrokeEventTokenizer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("StrokeEventTokenizer requires torch")
