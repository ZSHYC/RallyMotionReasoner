from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import functional as F


def masked_bce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def masked_smooth_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def compute_loss(outputs: dict[str, Tensor], batch: dict[str, Any], loss_weights: dict[str, float] | None = None) -> Tensor:
    weights = loss_weights or {}
    loss = float(weights.get("evidence", 2.0)) * masked_bce(outputs["evidence_logits"], batch["evidence"], batch["node_mask"])
    loss = loss + float(weights.get("key_action", 2.0)) * masked_bce(outputs["key_action_logits"], batch["key"], batch["node_mask"])
    if "ball_xy" in outputs and float(weights.get("ball_xy", 0.0)) > 0:
        loss = loss + float(weights.get("ball_xy", 0.0)) * masked_smooth_l1(outputs["ball_xy"], batch["ball_xy"], batch["ball_mask"])
    if "ball_visible_logits" in outputs and float(weights.get("ball_visible", 0.0)) > 0:
        loss = loss + float(weights.get("ball_visible", 0.0)) * masked_bce(outputs["ball_visible_logits"], batch["ball_visible"], batch["ball_mask"])
    if "contact_frame" in outputs and float(weights.get("contact_frame", 0.0)) > 0:
        loss = loss + float(weights.get("contact_frame", 0.0)) * masked_smooth_l1(outputs["contact_frame"], batch["contact_frame"], batch["contact_mask"])
    for name, labels in batch["labels"].items():
        weight = float(weights.get(name, 1.0 if name in {"level_1", "level_2", "level_3"} else 0.5))
        loss = loss + weight * F.cross_entropy(outputs[f"{name}_logits"], labels)
    return loss
