"""Region-infer event experts and motion-aware cross-modal fusion.

The two expert classes intentionally preserve the published
``tennis-region-infer`` parameter names so its expert checkpoints can be used
without copying that repository into TennisVAR. ``RegionFusionEventModel`` is
the trainable upgrade: it exchanges information between trajectory motion
tokens and five-view visual tokens before event readout.
"""

from __future__ import annotations

import math
from typing import Any

from .labels import ATTRIBUTE_FIELDS

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None


if nn is not None:

    class _ResidualConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(64, 64, kernel_size=5, padding=2)
            self.dropout = nn.Dropout(0.25)
            self.norm = nn.LayerNorm(64)

        def forward(self, values: Tensor) -> Tensor:
            residual = values
            values = self.dropout(F.gelu(self.conv(values))) + residual
            return self.norm(values.transpose(1, 2)).transpose(1, 2)


    class _Heads(nn.Module):
        def _init_heads(self) -> None:
            self.eventness_head = nn.Sequential(
                nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, 1)
            )
            self.type_head = nn.Sequential(
                nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, 2)
            )

        def _heads(self, pooled: Tensor) -> dict[str, Tensor]:
            return {
                "eventness_logit": self.eventness_head(pooled).squeeze(-1),
                "type_logits": self.type_head(pooled),
            }


    class TrajectoryExpert(_Heads):
        """Published 11-D motion expert from tennis-region-infer."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(779, 64)
            self.input_norm = nn.LayerNorm(64)
            self.input_dropout = nn.Dropout(0.25)
            self.convs = nn.ModuleList(_ResidualConv() for _ in range(3))
            self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
            self._init_heads()

        def encode(self, trajectory: Tensor) -> tuple[Tensor, Tensor]:
            if trajectory.ndim != 3 or trajectory.shape[-1] != 11:
                raise ValueError("trajectory must have shape [B,T,11]")
            valid = trajectory[..., 9] > 0.5
            values = F.linear(trajectory, self.proj.weight[:, :11], self.proj.bias)
            values = self.input_dropout(F.gelu(self.input_norm(values))).transpose(1, 2)
            for block in self.convs:
                values = block(values)
            values, _ = self.gru(values.transpose(1, 2))
            mask = valid[..., None]
            temporal_mean = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            temporal_max = values.masked_fill(~mask, -torch.inf).amax(dim=1)
            temporal_max = torch.where(torch.isfinite(temporal_max), temporal_max, torch.zeros_like(temporal_max))
            center = values[:, values.shape[1] // 2]
            return values, torch.cat((center, temporal_mean, temporal_max), dim=-1)

        def forward(self, trajectory: Tensor) -> dict[str, Tensor]:
            _, pooled = self.encode(trajectory)
            return self._heads(pooled)


    class VisualExpert(_Heads):
        """Published five-view region expert from tennis-region-infer."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(768, 64)
            self.input_norm = nn.LayerNorm(64)
            self.input_dropout = nn.Dropout(0.25)
            self.position_proj = nn.Linear(2, 64, bias=False)
            self.register_buffer(
                "tile_positions",
                torch.tensor(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))),
                persistent=False,
            )
            self.convs = nn.ModuleList(_ResidualConv() for _ in range(3))
            self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
            self._init_heads()

        def encode(self, visual: Tensor) -> tuple[Tensor, Tensor]:
            if visual.ndim != 3 or visual.shape[-1] != 3840:
                raise ValueError("visual must have shape [B,T,3840]")
            tokens = visual.reshape(*visual.shape[:2], 5, 768).to(self.proj.weight.dtype)
            projected = self.input_norm(self.proj(tokens))
            batch_size, steps, regions, hidden = projected.shape
            values = projected.permute(0, 2, 3, 1).reshape(batch_size * regions, hidden, steps)
            for block in self.convs:
                values = block(values)
            projected = values.reshape(batch_size, regions, hidden, steps).permute(0, 3, 1, 2)
            global_token = projected[:, :, 0]
            keys = projected[:, :, 1:] + self.position_proj(self.tile_positions)
            scores = (global_token.unsqueeze(2) * keys).sum(dim=-1) / math.sqrt(hidden)
            weights = scores.softmax(dim=-1)
            values = global_token + (weights.unsqueeze(-1) * keys).sum(dim=2)
            values = self.input_dropout(F.gelu(self.input_norm(values))).transpose(1, 2)
            values, _ = self.gru(values.transpose(1, 2))
            pooled = torch.cat(
                (values[:, values.shape[1] // 2], values.mean(dim=1), values.amax(dim=1)), dim=-1
            )
            return values, pooled

        def forward(self, visual: Tensor) -> dict[str, Tensor]:
            _, pooled = self.encode(visual)
            return self._heads(pooled)


    class RegionFusionEventModel(nn.Module):
        """Motion-to-region cross-attention event detector.

        The trajectory stream is an explicit motion token sequence. Two
        cross-attention directions let visual regions query motion and motion
        query visual context before the hit/bounce heads. This is a new model
        contract and therefore needs a checkpoint trained for this structure.
        """

        def __init__(
            self,
            attention_heads: int = 4,
            dropout: float = 0.1,
            attribute_maps: dict[str, dict[str, int]] | None = None,
        ) -> None:
            super().__init__()
            self.trajectory = TrajectoryExpert()
            self.visual = VisualExpert()
            self.motion_to_region = nn.MultiheadAttention(128, attention_heads, dropout=dropout, batch_first=True)
            self.region_to_motion = nn.MultiheadAttention(128, attention_heads, dropout=dropout, batch_first=True)
            self.fusion_norm = nn.LayerNorm(512)
            self.fusion = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout))
            self.eventness_head = nn.Linear(256, 1)
            self.type_head = nn.Linear(256, 2)
            self.attribute_heads = nn.ModuleDict(
                {
                    field: nn.Linear(256, len(attribute_maps[field]))
                    for field in ATTRIBUTE_FIELDS
                    if attribute_maps and attribute_maps.get(field)
                }
            )

        def forward(self, trajectory: Tensor, visual: Tensor) -> dict[str, Tensor]:
            motion_tokens, _ = self.trajectory.encode(trajectory)
            region_tokens, _ = self.visual.encode(visual)
            motion_context, _ = self.motion_to_region(motion_tokens, region_tokens, region_tokens, need_weights=False)
            region_context, _ = self.region_to_motion(region_tokens, motion_tokens, motion_tokens, need_weights=False)
            fused = torch.cat(
                [
                    motion_tokens.mean(dim=1),
                    region_tokens.mean(dim=1),
                    (motion_tokens + motion_context).mean(dim=1),
                    (region_tokens + region_context).mean(dim=1),
                ],
                dim=-1,
            )
            fused = self.fusion(self.fusion_norm(fused))
            output = {
                "eventness_logit": self.eventness_head(fused).squeeze(-1),
                "type_logits": self.type_head(fused),
            }
            output.update({f"{field}_logits": head(fused) for field, head in self.attribute_heads.items()})
            return output

else:

    class TrajectoryExpert:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("region event models require torch")

    class VisualExpert(TrajectoryExpert):  # type: ignore[no-redef]
        pass

    class RegionFusionEventModel(TrajectoryExpert):  # type: ignore[no-redef]
        pass
