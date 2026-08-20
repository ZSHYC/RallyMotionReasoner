from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    F = None
    Tensor = Any
    nn = None

from .labels import ATTRIBUTE_FIELDS, MISSING_INDEX


@dataclass(frozen=True)
class EventModelConfig:
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    temporal_kernel: int = 5

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


if nn is not None:

    class TemporalResidualBlock(nn.Module):
        def __init__(self, hidden_dim: int, kernel: int, dropout: float, dilation: int) -> None:
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, 1),
                nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, value: Tensor) -> Tensor:
            update = self.net(value.transpose(1, 2)).transpose(1, 2)
            return self.norm(value + update)


    class EventParsingModule(nn.Module):
        """Framewise event detector over frozen visual, motion and ball features."""

        def __init__(self, config: EventModelConfig, attribute_maps: dict[str, dict[str, int]]) -> None:
            super().__init__()
            if config.hidden_dim % config.num_heads:
                raise ValueError("hidden_dim must be divisible by num_heads")
            self.config = config
            self.attribute_maps = attribute_maps
            self.input = nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.GELU())
            self.local = nn.ModuleList(
                TemporalResidualBlock(config.hidden_dim, config.temporal_kernel, config.dropout, 2**index)
                for index in range(config.num_layers)
            )
            layer = nn.TransformerEncoderLayer(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=max(1, config.num_layers // 2))
            self.event_head = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1))
            self.attribute_heads = nn.ModuleDict(
                {field: nn.Linear(config.hidden_dim, len(attribute_maps.get(field, {}))) for field in ATTRIBUTE_FIELDS if attribute_maps.get(field)}
            )

        def forward(self, features: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
            if features.ndim != 3:
                raise ValueError("features must have shape [batch, frames, dim]")
            hidden = self.input(features)
            for block in self.local:
                hidden = block(hidden)
            hidden = self.temporal(hidden, src_key_padding_mask=None if mask is None else ~mask.bool())
            output = {"event_logits": self.event_head(hidden).squeeze(-1), "frame_tokens": hidden}
            output.update({f"{field}_logits": head(hidden) for field, head in self.attribute_heads.items()})
            return output


    def focal_bce_with_logits(logits: Tensor, targets: Tensor, mask: Tensor | None = None, gamma: float = 2.0) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        pt = targets * probability + (1.0 - targets) * (1.0 - probability)
        loss = loss * (1.0 - pt).pow(gamma)
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp_min(1.0)
        return loss.mean()


    def event_detector_loss(
        outputs: dict[str, Tensor],
        event_targets: Tensor,
        attribute_targets: dict[str, Tensor],
        mask: Tensor | None = None,
        *,
        attribute_weight: float = 0.5,
    ) -> dict[str, Tensor]:
        event_loss = focal_bce_with_logits(outputs["event_logits"], event_targets, mask)
        attribute_losses: list[Tensor] = []
        for field, target in attribute_targets.items():
            key = f"{field}_logits"
            if key not in outputs:
                continue
            flat_target = target.reshape(-1)
            valid = flat_target.ne(MISSING_INDEX)
            if valid.any():
                logits = outputs[key].reshape(-1, outputs[key].shape[-1])[valid]
                attribute_losses.append(F.cross_entropy(logits, flat_target[valid]))
        attribute_loss = torch.stack(attribute_losses).mean() if attribute_losses else event_loss.new_zeros(())
        return {"loss": event_loss + attribute_weight * attribute_loss, "event_loss": event_loss, "attribute_loss": attribute_loss}

else:

    class EventParsingModule:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("EventParsingModule requires torch")
