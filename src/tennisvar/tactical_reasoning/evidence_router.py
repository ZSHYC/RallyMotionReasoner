from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover
    torch = None
    Tensor = object
    nn = None

from .evidence_head import EvidenceGroundingHead


@dataclass
class EvidenceRouterOutput:
    evidence_logits: Tensor
    key_action_logits: Tensor
    routed_token: Tensor
    evidence_weights: Tensor


if nn is not None:

    class EvidenceRouter(nn.Module):
        """Evidence-first router for thinking-with-video style reasoning."""

        def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.head = EvidenceGroundingHead(hidden_dim, dropout=dropout)
            self.route_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid(),
            )
            self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU())

        def _masked_softmax(self, logits: Tensor, mask: Tensor | None) -> Tensor:
            if mask is None:
                return torch.softmax(logits, dim=-1)
            masked = logits.masked_fill(~mask.bool(), -1e4)
            weights = torch.softmax(masked, dim=-1)
            return weights * mask.float()

        def forward(
            self,
            stroke_tokens: Tensor,
            graph_token: Tensor,
            question_token: Tensor,
            node_mask: Tensor | None = None,
            *,
            enabled: bool = True,
        ) -> EvidenceRouterOutput:
            scores = self.head(stroke_tokens, question_token)
            weights = self._masked_softmax(scores.evidence_logits, node_mask)
            evidence_context = torch.bmm(weights.unsqueeze(1), stroke_tokens).squeeze(1)
            if not enabled:
                return EvidenceRouterOutput(scores.evidence_logits, scores.key_action_logits, graph_token, weights)
            joint = torch.cat([graph_token, evidence_context, question_token], dim=-1)
            gate = self.route_gate(joint)
            routed = self.out(torch.cat([gate * graph_token, (1.0 - gate) * evidence_context, question_token], dim=-1))
            return EvidenceRouterOutput(scores.evidence_logits, scores.key_action_logits, routed, weights)

else:

    class EvidenceRouter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("EvidenceRouter requires torch to be installed.")
