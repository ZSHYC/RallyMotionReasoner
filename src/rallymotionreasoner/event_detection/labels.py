from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rallymotionreasoner.stroke_labels import parse_label

ATTRIBUTE_FIELDS = ("hitter", "court_zone", "phase", "hand", "technique", "direction", "approach", "outcome")
MISSING_INDEX = -100


def build_attribute_maps(labels: Iterable[str]) -> dict[str, dict[str, int]]:
    values: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        parsed = parse_label(str(label))
        for field in ATTRIBUTE_FIELDS:
            value = parsed.get(field)
            if value is not None:
                values[field].add(str(value))
    return {field: {value: index for index, value in enumerate(sorted(values[field]))} for field in ATTRIBUTE_FIELDS}


def encode_attribute(field: str, value: Any, maps: dict[str, dict[str, int]]) -> int:
    if value is None or str(value) in {"", "-"}:
        return MISSING_INDEX
    return int(maps.get(field, {}).get(str(value), MISSING_INDEX))


def decode_attribute(field: str, index: int, maps: dict[str, dict[str, int]]) -> str | None:
    inverse = {value: key for key, value in maps.get(field, {}).items()}
    return inverse.get(int(index))


def encoded_event_attributes(label: str, maps: dict[str, dict[str, int]]) -> dict[str, int]:
    parsed = parse_label(str(label))
    return {field: encode_attribute(field, parsed.get(field), maps) for field in ATTRIBUTE_FIELDS}
