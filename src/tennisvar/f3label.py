from __future__ import annotations

from typing import Any

LABEL_FIELDS = [
    "hitter",
    "court_zone",
    "phase",
    "hand",
    "technique",
    "direction",
    "approach_token",
    "outcome_label",
]


def none_if_dash(value: str | None) -> str | None:
    if value is None or value == "-" or value == "":
        return None
    return value


def parse_label(label: str, outcome: str | None = None) -> dict[str, Any]:
    parts = label.split("_")
    parsed: dict[str, Any] = {"raw_label": label, "parse_ok": len(parts) == 8}
    for field, value in zip(LABEL_FIELDS, parts):
        parsed[field] = none_if_dash(value)
    if len(parts) != 8:
        parsed["raw_parts"] = parts
    approach_token = parsed.get("approach_token")
    parsed["approach"] = bool(approach_token and approach_token != "-")
    parsed["outcome"] = outcome or parsed.get("outcome_label")
    parsed["is_terminal"] = parsed.get("outcome") in {"winner", "forced-err", "unforced-err"}
    return parsed


def stroke_doc(shot: dict[str, Any]) -> str:
    parsed = shot.get("parsed") or {}
    fields = [
        f"shot_id={shot.get('shot_id')}",
        f"frame={shot.get('frame')}",
        f"hitter={parsed.get('hitter')}",
        f"court_zone={parsed.get('court_zone')}",
        f"phase={parsed.get('phase')}",
        f"hand={parsed.get('hand')}",
        f"technique={parsed.get('technique')}",
        f"direction={parsed.get('direction')}",
        f"approach={parsed.get('approach')}",
        f"outcome={parsed.get('outcome')}",
        f"label={shot.get('label')}",
    ]
    return "; ".join(str(x) for x in fields if x is not None)
