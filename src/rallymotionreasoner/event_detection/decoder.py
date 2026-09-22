from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecodedEvent:
    frame: int
    time_sec: float
    confidence: float
    attributes: dict[str, str | None]
    attribute_confidence: dict[str, float]
    source: str = "motion_region"
    event_type: str = "hit"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
