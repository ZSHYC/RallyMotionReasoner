"""Region-infer event experts and motion-aware cross-modal fusion.

The two expert classes intentionally preserve the published
``tennis-region-infer`` parameter names so its expert checkpoints can be used
without copying that repository into RallyMotionReasoner. ``MotionRegionEventModel`` is
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
            # The released expert keeps 779 projection weights; only 11 trajectory inputs are active.
            self.proj = nn.Linear(779, 64)
            self.input_norm = nn.LayerNorm(64)
            self.input_dropout = nn.Dropout(0.25)
            self.convs = nn.ModuleList(_ResidualConv() for _ in range(3))
            self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
            self._init_heads()

        def _encode_sequence(self, trajectory: Tensor) -> Tensor:
            values = F.linear(trajectory, self.proj.weight[:, :11], self.proj.bias)
            values = self.input_dropout(F.gelu(self.input_norm(values))).transpose(1, 2)
            for block in self.convs:
                values = block(values)
            values, _ = self.gru(values.transpose(1, 2))
            return values

        @staticmethod
        def _pool(values: Tensor, valid: Tensor, center: int) -> Tensor:
            mask = valid[..., None]
            temporal_mean = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            temporal_max = values.masked_fill(~mask, -torch.inf).amax(dim=1)
            temporal_max = torch.where(
                torch.isfinite(temporal_max), temporal_max, torch.zeros_like(temporal_max)
            )
            return torch.cat((values[:, center], temporal_mean, temporal_max), dim=-1)

        def encode(self, trajectory: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
            if trajectory.ndim != 3 or trajectory.shape[-1] != 11:
                raise ValueError("trajectory must have shape [B,T,11]")
            valid = trajectory[..., 9] > 0.5
            if padding_mask is None:
                values = self._encode_sequence(trajectory)
                return values, self._pool(values, valid, values.shape[1] // 2)
            if padding_mask.shape != trajectory.shape[:2]:
                raise ValueError("padding_mask must have shape [B,T]")
            padding_mask = padding_mask.bool()
            if bool(padding_mask.all()):
                values = self._encode_sequence(trajectory)
                return values, self._pool(values, valid, values.shape[1] // 2)
            center = trajectory.shape[1] // 2
            if not bool(padding_mask[:, center].all()):
                raise ValueError("padding_mask must include the center timestep")
            values = trajectory.new_zeros((*trajectory.shape[:2], 128))
            pooled = values.new_empty((trajectory.shape[0], 384))
            complete = padding_mask.all(dim=1)
            if bool(complete.any()):
                encoded = self._encode_sequence(trajectory[complete])
                values[complete] = encoded
                pooled[complete] = self._pool(encoded, valid[complete], center)
            for index in (~complete).nonzero().flatten().tolist():
                positions = padding_mask[index].nonzero().flatten()
                encoded = self._encode_sequence(trajectory[index : index + 1, positions])
                values[index, positions] = encoded[0]
                local_center = int((positions < center).sum())
                pooled[index] = self._pool(
                    encoded, valid[index : index + 1, positions], local_center
                )[0]
            return values, pooled

        def forward(self, trajectory: Tensor, padding_mask: Tensor | None = None) -> dict[str, Tensor]:
            _, pooled = self.encode(trajectory, padding_mask)
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

        def _encode_sequence(self, visual: Tensor) -> Tensor:
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
            return values

        @staticmethod
        def _pool(values: Tensor, center: int) -> Tensor:
            return torch.cat((values[:, center], values.mean(dim=1), values.amax(dim=1)), dim=-1)

        def encode(self, visual: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
            if visual.ndim != 3 or visual.shape[-1] != 3840:
                raise ValueError("visual must have shape [B,T,3840]")
            if padding_mask is None:
                values = self._encode_sequence(visual)
                return values, self._pool(values, values.shape[1] // 2)
            if padding_mask.shape != visual.shape[:2]:
                raise ValueError("padding_mask must have shape [B,T]")
            padding_mask = padding_mask.bool()
            if bool(padding_mask.all()):
                values = self._encode_sequence(visual)
                return values, self._pool(values, values.shape[1] // 2)
            center = visual.shape[1] // 2
            if not bool(padding_mask[:, center].all()):
                raise ValueError("padding_mask must include the center timestep")
            values = visual.new_zeros((*visual.shape[:2], 128), dtype=self.proj.weight.dtype)
            pooled = values.new_empty((visual.shape[0], 384))
            complete = padding_mask.all(dim=1)
            if bool(complete.any()):
                encoded = self._encode_sequence(visual[complete])
                values[complete] = encoded
                pooled[complete] = self._pool(encoded, center)
            for index in (~complete).nonzero().flatten().tolist():
                positions = padding_mask[index].nonzero().flatten()
                encoded = self._encode_sequence(visual[index : index + 1, positions])
                values[index, positions] = encoded[0]
                local_center = int((positions < center).sum())
                pooled[index] = self._pool(encoded, local_center)[0]
            return values, pooled

        def forward(self, visual: Tensor, padding_mask: Tensor | None = None) -> dict[str, Tensor]:
            _, pooled = self.encode(visual, padding_mask)
            return self._heads(pooled)


    class MotionRegionEventModel(nn.Module):
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

        def forward(
            self, trajectory: Tensor, visual: Tensor, visual_mask: Tensor | None = None
        ) -> dict[str, Tensor]:
            if trajectory.ndim != 3 or trajectory.shape[-1] != 11 or not trajectory.shape[1]:
                raise ValueError("trajectory must have non-empty shape [B,T,11]")
            if visual.ndim != 3 or visual.shape[-1] != 3840 or not visual.shape[1] or visual.shape[0] != trajectory.shape[0]:
                raise ValueError("visual must have non-empty shape [B,T,3840] with the same batch size")
            motion_mask = trajectory[..., 9] > 0.5
            if visual_mask is None:
                visual_mask = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
            if visual_mask.shape != visual.shape[:2]:
                raise ValueError("visual_mask must have shape [B,T]")
            visual_mask = visual_mask.bool()
            motion_tokens, _ = self.trajectory.encode(trajectory.masked_fill(~motion_mask[..., None], 0))
            region_tokens, _ = self.visual.encode(visual.masked_fill(~visual_mask[..., None], 0))
            motion_tokens = motion_tokens.masked_fill(~motion_mask[..., None], 0)
            region_tokens = region_tokens.masked_fill(~visual_mask[..., None], 0)
            # Empty streams expose one zero key to avoid all-masked attention.
            safe_motion, safe_visual = motion_mask.clone(), visual_mask.clone()
            safe_motion[~motion_mask.any(dim=1), 0] = True
            safe_visual[~visual_mask.any(dim=1), 0] = True
            motion_context, _ = self.motion_to_region(
                motion_tokens, region_tokens, region_tokens, key_padding_mask=~safe_visual, need_weights=False
            )
            region_context, _ = self.region_to_motion(
                region_tokens, motion_tokens, motion_tokens, key_padding_mask=~safe_motion, need_weights=False
            )
            motion_context = motion_context * visual_mask.any(dim=1)[:, None, None]
            region_context = region_context * motion_mask.any(dim=1)[:, None, None]

            def pool(tokens: Tensor, mask: Tensor) -> Tensor:
                return (tokens * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

            fused = torch.cat(
                [
                    pool(motion_tokens, motion_mask),
                    pool(region_tokens, visual_mask),
                    pool(motion_tokens + motion_context, motion_mask),
                    pool(region_tokens + region_context, visual_mask),
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

    class MotionRegionEventModel(TrajectoryExpert):  # type: ignore[no-redef]
        pass
