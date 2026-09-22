from __future__ import annotations

import json
from typing import Any

REQUIRED_SCHEMA = [
    "answer",
    "answer_type",
    "level_1",
    "level_2",
    "level_3",
    "evidence_shot_ids",
    "key_action_shot_ids",
    "evidence_frames",
    "observed_effect",
    "causal_strength",
    "answerability",
    "explanation",
]

# New temporal key-action field is optional for backwards-compatible reads.
OPTIONAL_SCHEMA = ["key_action_frames"]

STRICT_ANSWER_FIELDS = [
    "answer",
    "level1_stage",
    "level2_family",
    "level3_tactic",
    "evidence_shot_ids",
    "key_action_shot_ids",
    "evidence_frames",
    "observed_effect",
    "causal_strength",
    "answerability",
    "explanation",
]

ANSWER_TYPES = {"short_answer", "free_form", "verification", "counterfactual_limited"}
ANSWERABILITY = {"answerable", "partially_answerable", "unanswerable"}
OBSERVED_EFFECTS = {"winner", "forced_error", "unforced_error", "continuation", "countered", "unknown"}
CAUSAL_STRENGTHS = {"verified", "plausible_rule_based", "not_verified", "insufficient"}

FACTUAL_QA_TYPES = {
    "shot_count",
    "serve_direction",
    "serve_outcome",
    "final_outcome",
    "final_hitter",
    "return_direction",
    "return_hand",
    "specific_shot_attribute",
}
VERIFICATION_QA_TYPES = {"claim_verification"}
COUNTERFACTUAL_QA_TYPES = {
    "decision_quality_limited",
    "decision_under_uncertainty",
    "risk_comparison_in_context",
    "tactical_tradeoff",
}

KNOWN_TACTICS = {
    "approach_plus_first_volley",
    "approach_transition",
    "chip_and_charge",
    "direct_return_attack",
    "direct_serve_pressure",
    "direction_change",
    "drop_shot_attempt",
    "lob_against_net",
    "lob_attempt",
    "lob_to_smash",
    "multi_net_attack",
    "net_attack",
    "no_primary_tactic",
    "passing_attempt",
    "repeated_pattern",
    "return_neutralization",
    "return_plus_one_attack",
    "serve_and_volley",
    "serve_plus_one_attack",
    "serve_plus_one_breakdown",
    "slice_transition",
    "successful_pass",
    "sustained_exchange",
}


def validate_prediction_payload(value: Any) -> list[str]:
    """Return exact schema violations for a raw model prediction.

    This deliberately validates before normalization. Silently filling missing
    model fields would make JSON-validity and answerability metrics misleading.
    """
    if not isinstance(value, dict):
        return ["prediction must be a JSON object"]
    errors: list[str] = []
    missing = [field for field in REQUIRED_SCHEMA if field not in value]
    if missing:
        errors.append(f"missing fields: {missing}")
    if "answer" in value and not isinstance(value["answer"], str):
        errors.append("answer must be a string")
    if "explanation" in value and not isinstance(value["explanation"], str):
        errors.append("explanation must be a string")
    if value.get("answer_type") not in ANSWER_TYPES:
        errors.append(f"invalid answer_type: {value.get('answer_type')!r}")
    if value.get("answerability") not in ANSWERABILITY:
        errors.append(f"invalid answerability: {value.get('answerability')!r}")
    if value.get("observed_effect") not in OBSERVED_EFFECTS:
        errors.append(f"invalid observed_effect: {value.get('observed_effect')!r}")
    if value.get("causal_strength") not in CAUSAL_STRENGTHS:
        errors.append(f"invalid causal_strength: {value.get('causal_strength')!r}")
    for field in ("level_1", "level_2", "level_3"):
        if field in value and value[field] is not None and not isinstance(value[field], str):
            errors.append(f"{field} must be a string or null")
    for field in ("evidence_shot_ids", "key_action_shot_ids", "evidence_frames", "key_action_frames"):
        if field not in value:
            continue
        items = value.get(field)
        if not isinstance(items, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in items):
            errors.append(f"{field} must be an integer array")
    if isinstance(value.get("evidence_shot_ids"), list) and isinstance(value.get("key_action_shot_ids"), list):
        evidence_ids = set(value["evidence_shot_ids"])
        if not set(value["key_action_shot_ids"]).issubset(evidence_ids):
            errors.append("key_action_shot_ids must be a subset of evidence_shot_ids")
    if isinstance(value.get("evidence_frames"), list) and any(frame < 0 for frame in value["evidence_frames"]):
        errors.append("evidence_frames must be non-negative")
    return errors


def to_int_list(values: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(values, list):
        return out
    for value in values:
        if isinstance(value, int):
            out.append(value)
        elif str(value).isdigit():
            out.append(int(value))
    return out


def compact_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_effect(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"key_action_winner", "sequence_completion_winner", "winner"}:
        return "winner"
    if text in {"forced_error", "forced_err", "opponent_forced_error", "induced_forced_error"}:
        return "forced_error"
    if text in {"execution_forced_error"}:
        return "forced_error"
    if text in {"unforced_error", "unforced_err", "execution_unforced_error"}:
        return "unforced_error"
    if text in {"opponent_winner", "countered_by_opponent_winner", "countered"}:
        return "countered"
    if text in {"in", "continuation", "continuation_or_in", "completed_no_terminal_effect", "no_verified_effect"}:
        return "continuation"
    if "winner" in text:
        return "winner"
    if "forced" in text and "unforced" not in text:
        return "forced_error"
    if "unforced" in text:
        return "unforced_error"
    if "counter" in text or "opponent_winner" in text:
        return "countered"
    if "continu" in text or text == "none":
        return "continuation"
    return "unknown"


def normalize_causal_strength(value: Any, *, answerability: str | None = None) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if answerability in {"partially_answerable", "unanswerable"} and text in {"", "none", "not_counterfactual"}:
        return "insufficient"
    if text in CAUSAL_STRENGTHS:
        return text
    if text in {"event_supported", "direct_event", "supported", "oracle", "gold"}:
        return "verified"
    if text in {"plausible", "rule_based", "plausible_rule"}:
        return "plausible_rule_based"
    if text in {"none", "not_counterfactual", "unknown", ""}:
        return "insufficient"
    return "insufficient"


def answer_type_for_qa_type(qa_type: str | None) -> str:
    qt = str(qa_type or "")
    if qt in FACTUAL_QA_TYPES:
        return "short_answer"
    if qt in VERIFICATION_QA_TYPES:
        return "verification"
    if qt in COUNTERFACTUAL_QA_TYPES:
        return "counterfactual_limited"
    return "free_form"


def answerability_for_qa_type(qa_type: str | None, current: Any = None) -> str:
    text = str(current or "").strip()
    if text in ANSWERABILITY:
        return text
    return "partially_answerable" if str(qa_type or "") in COUNTERFACTUAL_QA_TYPES else "answerable"


def convert_answer(answer: dict[str, Any] | None, *, qa_type: str | None = None) -> dict[str, Any]:
    answer = answer or {}
    answerability = answerability_for_qa_type(qa_type, answer.get("answerability"))
    level_1 = answer.get("level_1", answer.get("level1_stage"))
    level_2 = answer.get("level_2", answer.get("level2_family"))
    level_3 = answer.get("level_3", answer.get("level3_tactic"))
    return {
        "answer": str(answer.get("answer") or ""),
        "answer_type": answer.get("answer_type") if answer.get("answer_type") in ANSWER_TYPES else answer_type_for_qa_type(qa_type),
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
        "evidence_shot_ids": to_int_list(answer.get("evidence_shot_ids")),
        "key_action_shot_ids": to_int_list(answer.get("key_action_shot_ids")),
        "evidence_frames": to_int_list(answer.get("evidence_frames")),
        "key_action_frames": to_int_list(answer.get("key_action_frames")),
        "observed_effect": normalize_effect(answer.get("observed_effect")),
        "causal_strength": normalize_causal_strength(answer.get("causal_strength"), answerability=answerability),
        "answerability": answerability,
        "explanation": str(answer.get("explanation") or ""),
    }


def answer_payload(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("gold_answer"), dict):
        return convert_answer(row["gold_answer"], qa_type=row.get("qa_type"))
    if isinstance(row.get("answer"), dict):
        return convert_answer(row["answer"], qa_type=row.get("qa_type"))
    return convert_answer({}, qa_type=row.get("qa_type"))


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    gold = answer_payload(row)
    out["gold_answer"] = gold
    out.setdefault("answer", gold)
    out["schema_version"] = "eval"
    return out


def prediction_to(pred: dict[str, Any]) -> dict[str, Any]:
    if pred.get("parse_error"):
        out = {"qa_id": pred.get("qa_id"), "rally_id": pred.get("rally_id"), "parse_error": pred.get("parse_error")}
        for field in REQUIRED_SCHEMA + OPTIONAL_SCHEMA:
            out.setdefault(field, None if field not in {"evidence_shot_ids", "key_action_shot_ids", "evidence_frames", "key_action_frames"} else [])
        return out
    out = convert_answer(pred, qa_type=pred.get("qa_type"))
    out["qa_id"] = pred.get("qa_id")
    if pred.get("rally_id"):
        out["rally_id"] = pred.get("rally_id")
    return out
