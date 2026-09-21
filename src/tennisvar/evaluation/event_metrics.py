from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from tennisvar.event_parsing.labels import ATTRIBUTE_FIELDS
from tennisvar.evaluation.matching import optimal_temporal_matching


def one_to_one_match(predicted: list[int], gold: list[int], tolerance: int) -> list[tuple[int, int]]:
    return optimal_temporal_matching(predicted, gold, tolerance)


def average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(int(value > 0) for value in labels)
    if positives == 0:
        return 0.0
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda item: scores[item], reverse=True), start=1):
        if labels[index] > 0:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _macro_f1(gold: list[str], predicted: list[str]) -> float:
    labels = sorted(set(gold) | set(predicted))
    values: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values) if values else 0.0


def _canonical_attribute(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _aggregate_event_records(
    records: list[dict[str, Any]], tolerances: tuple[int, ...]
) -> dict[str, float]:
    totals = {tol: {"tp": 0, "pred": 0, "gold": 0} for tol in tolerances}
    count_errors: list[int] = []
    frame_errors: list[int] = []
    attribute_correct: defaultdict[str, int] = defaultdict(int)
    attribute_total: defaultdict[str, int] = defaultdict(int)
    attribute_gold: defaultdict[str, list[str]] = defaultdict(list)
    attribute_predicted: defaultdict[str, list[str]] = defaultdict(list)
    complete_correct = complete_total = 0
    aps: list[float] = []
    empty = 0
    for record in records:
        predicted = [int(value) for value in record.get("predicted_frames") or []]
        gold = [int(value) for value in record.get("gold_frames") or []]
        count_errors.append(abs(len(predicted) - len(gold)))
        empty += int(not predicted)
        for tolerance in tolerances:
            matched = one_to_one_match(predicted, gold, tolerance)
            totals[tolerance]["tp"] += len(matched)
            totals[tolerance]["pred"] += len(predicted)
            totals[tolerance]["gold"] += len(gold)
        matched = one_to_one_match(predicted, gold, max(tolerances))
        pred_attrs = record.get("predicted_attributes") or []
        gold_attrs = record.get("gold_attributes") or []
        for pred_index, gold_index in matched:
            frame_errors.append(abs(predicted[pred_index] - gold[gold_index]))
            gold_row = gold_attrs[gold_index] if gold_index < len(gold_attrs) else {}
            pred_row = pred_attrs[pred_index] if pred_index < len(pred_attrs) else {}
            comparable = []
            for field in ATTRIBUTE_FIELDS:
                value = gold_row.get(field)
                if value is None:
                    continue
                predicted_value = pred_row.get(field)
                gold_value = _canonical_attribute(value)
                predicted_canonical = _canonical_attribute(predicted_value)
                attribute_total[field] += 1
                attribute_correct[field] += int(predicted_canonical == gold_value)
                attribute_gold[field].append(gold_value)
                attribute_predicted[field].append(predicted_canonical)
                comparable.append(predicted_canonical == gold_value)
            if comparable:
                complete_total += 1
                complete_correct += int(all(comparable))
        if record.get("frame_scores") is not None and record.get("frame_labels") is not None:
            aps.append(average_precision(list(record["frame_scores"]), list(record["frame_labels"])))
    metrics: dict[str, float] = {}
    for tolerance, values in totals.items():
        precision = values["tp"] / values["pred"] if values["pred"] else 0.0
        recall = values["tp"] / values["gold"] if values["gold"] else 0.0
        metrics[f"event_precision@{tolerance}"] = precision
        metrics[f"event_recall@{tolerance}"] = recall
        metrics[f"event_f1@{tolerance}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics["contact_ap"] = sum(aps) / len(aps) if aps else 0.0
    metrics["shot_count_mae"] = sum(count_errors) / len(count_errors) if count_errors else 0.0
    metrics["matched_frame_mae@16"] = sum(frame_errors) / len(frame_errors) if frame_errors else 0.0
    metrics["empty_prediction_rate"] = empty / len(records) if records else 0.0
    for field, total in attribute_total.items():
        metrics[f"attribute_{field}_accuracy@16"] = attribute_correct[field] / total if total else 0.0
        metrics[f"attribute_{field}_macro_f1@16"] = _macro_f1(attribute_gold[field], attribute_predicted[field])
    attribute_values = [metrics[f"attribute_{field}_accuracy@16"] for field in ATTRIBUTE_FIELDS if field in attribute_total]
    metrics["attribute_macro_accuracy@16"] = sum(attribute_values) / len(attribute_values) if attribute_values else 0.0
    metrics["attribute_complete_label_accuracy@16"] = complete_correct / complete_total if complete_total else 0.0
    return metrics


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def evaluate_event_records(
    records: list[dict[str, Any]],
    tolerances: tuple[int, ...] = (0, 2, 4, 8, 16),
    *,
    include_per_record: bool = False,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate F3ED with optional, reproducible formal-evaluation details."""
    overall = _aggregate_event_records(records, tolerances)
    report: dict[str, Any] = {"count": len(records), "overall": overall}
    if include_per_record:
        report["per_record"] = [
            {"rally_id": str(record.get("rally_id") or ""), **_aggregate_event_records([record], tolerances)}
            for record in records
        ]
    if bootstrap_samples > 0:
        if not records:
            report["bootstrap_95"] = {
                metric: {"estimate": value, "lower": value, "upper": value}
                for metric, value in overall.items()
            }
        else:
            rng = random.Random(bootstrap_seed)
            sampled_values = {metric: [] for metric in overall}
            for _ in range(bootstrap_samples):
                sample = [records[rng.randrange(len(records))] for _ in records]
                sample_metrics = _aggregate_event_records(sample, tolerances)
                for metric, value in sample_metrics.items():
                    sampled_values[metric].append(float(value))
            report["bootstrap_95"] = {
                metric: {
                    "estimate": float(overall[metric]),
                    "lower": _percentile(values, 0.025),
                    "upper": _percentile(values, 0.975),
                }
                for metric, values in sampled_values.items()
            }
        report["bootstrap_contract"] = {
            "aggregation": "corpus_micro_recomputed_per_resample",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        }
    return report
