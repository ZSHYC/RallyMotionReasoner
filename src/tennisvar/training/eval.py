from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from tennisvar.data.graph_qa import move_batch
from tennisvar.training.losses import compute_loss


def prf(pred: set[int], gold: set[int]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _visual_metric_counts(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, tuple[float, float]]:
    metrics: dict[str, float] = {
        "ball_pck@5": 0.0,
        "ball_pck@10": 0.0,
        "contact_acc@1": 0.0,
        "contact_acc@2": 0.0,
        "contact_acc@4": 0.0,
    }
    counts = {key: 0.0 for key in metrics}
    if "ball_xy" in outputs:
        dist_px = torch.linalg.norm((outputs["ball_xy"] - batch["ball_xy"]) * 224.0, dim=-1)
        mask = batch["ball_mask"].bool() & batch["node_mask"].bool()
        denom = float(mask.float().sum().detach().cpu())
        if denom > 0:
            for px in [5, 10]:
                key = f"ball_pck@{px}"
                metrics[key] += float(((dist_px <= px) & mask).float().sum().detach().cpu())
                counts[key] += denom
    if "contact_frame" in outputs:
        frame_scale = batch["frame_scale"].unsqueeze(-1).clamp_min(1.0)
        err = torch.abs(outputs["contact_frame"] - batch["contact_frame"]) * frame_scale
        mask = batch["contact_mask"].bool() & batch["node_mask"].bool()
        denom = float(mask.float().sum().detach().cpu())
        if denom > 0:
            for tol in [1, 2, 4]:
                key = f"contact_acc@{tol}"
                metrics[key] += float(((err <= tol) & mask).float().sum().detach().cpu())
                counts[key] += denom
    return {key: (metrics[key], counts[key]) for key in metrics}


@torch.no_grad()
def evaluate_internal(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    evidence_threshold: float = 0.45,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    total = 0
    evidence_f1 = 0.0
    key_acc = 0.0
    level_acc = {"level_1": 0.0, "level_2": 0.0, "level_3": 0.0}
    visual_sums = {"ball_pck@5": 0.0, "ball_pck@10": 0.0, "contact_acc@1": 0.0, "contact_acc@2": 0.0, "contact_acc@4": 0.0}
    visual_counts = {key: 0.0 for key in visual_sums}
    loss_sum = 0.0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        outputs = model(batch)
        loss_sum += float(compute_loss(outputs, batch, loss_weights).detach().cpu()) * len(batch["items"])
        visual = _visual_metric_counts(outputs, batch)
        for key, (value, count) in visual.items():
            visual_sums[key] += value
            visual_counts[key] += count
        ev_prob = torch.sigmoid(outputs["evidence_logits"]).detach().cpu()
        key_prob = torch.sigmoid(outputs["key_action_logits"]).detach().cpu()
        for i, item in enumerate(batch["items"]):
            n = len(item.shot_ids)
            pred_ev = {item.shot_ids[j] for j in range(n) if float(ev_prob[i, j]) >= evidence_threshold}
            if not pred_ev:
                pred_ev = {item.shot_ids[int(torch.argmax(ev_prob[i, :n]))]}
            gold_ev = {sid for sid, target in zip(item.shot_ids, item.evidence_targets) if target > 0.5}
            evidence_f1 += prf(pred_ev, gold_ev)[2]
            pred_key = {item.shot_ids[int(torch.argmax(key_prob[i, :n]))]}
            gold_key = {sid for sid, target in zip(item.shot_ids, item.key_targets) if target > 0.5}
            key_acc += 1.0 if pred_key == gold_key else 0.0
        for name in level_acc:
            pred = outputs[f"{name}_logits"].argmax(dim=-1)
            level_acc[name] += float((pred == batch["labels"][name]).float().sum().detach().cpu())
        total += len(batch["items"])
    return {
        "loss": loss_sum / total if total else 0.0,
        "evidence_f1": evidence_f1 / total if total else 0.0,
        "key_action_accuracy": key_acc / total if total else 0.0,
        **{key: visual_sums[key] / visual_counts[key] if visual_counts[key] else 0.0 for key in visual_sums},
        **{f"{name}_accuracy": value / total if total else 0.0 for name, value in level_acc.items()},
    }
