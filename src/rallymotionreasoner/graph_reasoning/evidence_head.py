from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover - allows import on machines without torch.
    torch = None
    Tensor = object
    nn = None


@dataclass
class EvidenceHeadOutput:
    """Scores for evidence grounding over candidate stroke tokens."""

    evidence_logits: Tensor
    key_action_logits: Tensor
    frame_logits: Tensor | None = None


if nn is not None:

    class EvidenceGroundingHead(nn.Module):
        """Question-conditioned evidence and key-action selector.

        Inputs use shape [batch, num_strokes, hidden_dim] for stroke tokens and
        [batch, hidden_dim] for question tokens. The head returns independent
        multi-label evidence scores and key-action scores.
        """

        def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.question_proj = nn.Linear(hidden_dim, hidden_dim)
            self.stroke_proj = nn.Linear(hidden_dim, hidden_dim)
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.evidence_scorer = nn.Linear(hidden_dim, 1)
            self.key_action_scorer = nn.Linear(hidden_dim, 1)

        def forward(self, stroke_tokens: Tensor, question_token: Tensor) -> EvidenceHeadOutput:
            question = self.question_proj(question_token).unsqueeze(1).expand_as(stroke_tokens)
            strokes = self.stroke_proj(stroke_tokens)
            fused = self.fusion(torch.cat([strokes, question, strokes * question], dim=-1))
            return EvidenceHeadOutput(
                evidence_logits=self.evidence_scorer(fused).squeeze(-1),
                key_action_logits=self.key_action_scorer(fused).squeeze(-1),
            )

else:

    class EvidenceGroundingHead:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("EvidenceGroundingHead requires torch to be installed.")
