from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .labels import ATTRIBUTE_FIELDS, decode_attribute


@dataclass(frozen=True)
class DecodedEvent:
    frame: int
    time_sec: float
    confidence: float
    attributes: dict[str, str | None]
    attribute_confidence: dict[str, float]
    source: str = "f3ed_predicted"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeakDecoder:
    threshold: float = 0.5
    min_separation_frames: int = 8
    top_k: int = 32

    def indices(self, scores: list[float], frames: list[int]) -> list[int]:
        if len(scores) != len(frames):
            raise ValueError("scores and frames length mismatch")
        local: list[int] = []
        for index, score in enumerate(scores):
            left = scores[index - 1] if index else float("-inf")
            right = scores[index + 1] if index + 1 < len(scores) else float("-inf")
            if score >= self.threshold and score >= left and score >= right:
                local.append(index)
        selected: list[int] = []
        for index in sorted(local, key=lambda item: (scores[item], -frames[item]), reverse=True):
            if all(abs(frames[index] - frames[other]) >= self.min_separation_frames for other in selected):
                selected.append(index)
            if len(selected) >= self.top_k:
                break
        return sorted(selected, key=lambda item: frames[item])

    def decode(
        self,
        scores: list[float],
        frames: list[int],
        attribute_probabilities: dict[str, list[list[float]]],
        attribute_maps: dict[str, dict[str, int]],
        *,
        fps: float,
    ) -> list[DecodedEvent]:
        events: list[DecodedEvent] = []
        for index in self.indices(scores, frames):
            attributes: dict[str, str | None] = {}
            confidence: dict[str, float] = {}
            for field in ATTRIBUTE_FIELDS:
                probabilities = attribute_probabilities.get(field, [])
                row = probabilities[index] if index < len(probabilities) else []
                if row:
                    best = max(range(len(row)), key=lambda item: row[item])
                    attributes[field] = decode_attribute(field, best, attribute_maps)
                    confidence[field] = float(row[best])
                else:
                    attributes[field] = None
                    confidence[field] = 0.0
            frame = int(frames[index])
            events.append(
                DecodedEvent(
                    frame=frame,
                    time_sec=round(frame / max(float(fps), 1e-6), 4),
                    confidence=float(scores[index]),
                    attributes=attributes,
                    attribute_confidence=confidence,
                )
            )
        return events
