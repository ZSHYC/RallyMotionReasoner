from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tennisvar.data.trace import load_source
from tennisvar.f3label import parse_label

from .labels import ATTRIBUTE_FIELDS, MISSING_INDEX, encode_attribute
from .targets import event_heatmap


def safe_name(value: str) -> str:
    return value.replace("/", "__")


def event_feature_file(root: Path, split: str, rally_id: str) -> Path:
    return Path(root) / split / f"{safe_name(rally_id)}.pt"


@dataclass
class EventSequence:
    rally_id: str
    features: Any
    frame_indices: Any
    event_targets: Any
    attribute_targets: dict[str, Any]
    provenance: dict[str, Any]


class EventSequenceDataset:
    def __init__(
        self,
        source_json: Path,
        feature_root: Path,
        split: str,
        attribute_maps: dict[str, dict[str, int]],
        *,
        target_radius: int = 4,
        limit: int = 0,
    ) -> None:
        import torch

        self.items: list[EventSequence] = []
        rows = load_source(Path(source_json))
        if limit > 0:
            rows = rows[:limit]
        for row in rows:
            rally_id = str(row["video"])
            path = event_feature_file(feature_root, split, rally_id)
            if not path.is_file():
                raise FileNotFoundError(f"missing event feature cache: {path}")
            cache = torch.load(path, map_location="cpu", weights_only=False)
            features = cache.get("features")
            frame_indices = cache.get("frame_indices")
            provenance = cache.get("provenance")
            if features is None or frame_indices is None or not provenance:
                raise ValueError(f"invalid event feature cache: {path}")
            expected_frames = int(row.get("num_frames") or 0)
            if (
                cache.get("schema") != "tennisvar.event_features.v1"
                or cache.get("rally_id") != rally_id
                or cache.get("split") != split
                or getattr(features, "ndim", None) != 2
                or int(features.shape[0]) != expected_frames
                or int(frame_indices.shape[0]) != expected_frames
                or int(provenance.get("feature_dim") or 0) != int(features.shape[-1])
                or not isinstance(provenance.get("weights_sha256"), str)
                or len(provenance["weights_sha256"]) != 64
            ):
                raise ValueError(f"event feature cache contract mismatch: {path}")
            if "tracknet8_real" in str(provenance.get("backend") or ""):
                tracknet_hash = provenance.get("tracknet_checkpoint_sha256")
                if not isinstance(tracknet_hash, str) or len(tracknet_hash) != 64:
                    raise ValueError(f"event feature cache lacks a valid TrackNet hash: {path}")
            frames = [int(value) for value in frame_indices.tolist()]
            if len(set(frames)) != len(frames) or frames != sorted(frames):
                raise ValueError(f"event feature frame indices must be unique and sorted: {path}")
            events = row.get("events") or []
            event_frames = [int(event["frame"]) for event in events]
            event_targets = torch.tensor(event_heatmap(frames, event_frames, target_radius, "gaussian"), dtype=torch.float32)
            attribute_targets = {field: torch.full((len(frames),), MISSING_INDEX, dtype=torch.long) for field in ATTRIBUTE_FIELDS}
            for event in events:
                event_frame = int(event["frame"])
                index = min(range(len(frames)), key=lambda item: abs(frames[item] - event_frame))
                parsed = parse_label(str(event.get("label") or ""), event.get("outcome"))
                for field in ATTRIBUTE_FIELDS:
                    attribute_targets[field][index] = encode_attribute(field, parsed.get(field), attribute_maps)
            self.items.append(EventSequence(rally_id, features.float(), torch.tensor(frames), event_targets, attribute_targets, provenance))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EventSequence:
        return self.items[index]


def collate_event_sequences(items: list[EventSequence]) -> dict[str, Any]:
    import torch

    if not items:
        raise ValueError("cannot collate an empty event batch")
    batch, maximum, dim = len(items), max(item.features.shape[0] for item in items), int(items[0].features.shape[-1])
    features = torch.zeros(batch, maximum, dim)
    frames = torch.zeros(batch, maximum, dtype=torch.long)
    event_targets = torch.zeros(batch, maximum)
    mask = torch.zeros(batch, maximum, dtype=torch.bool)
    attributes = {field: torch.full((batch, maximum), MISSING_INDEX, dtype=torch.long) for field in ATTRIBUTE_FIELDS}
    for batch_index, item in enumerate(items):
        length = int(item.features.shape[0])
        if int(item.features.shape[-1]) != dim:
            raise ValueError("event feature dimensions differ inside one batch")
        features[batch_index, :length] = item.features
        frames[batch_index, :length] = item.frame_indices
        event_targets[batch_index, :length] = item.event_targets
        mask[batch_index, :length] = True
        for field in ATTRIBUTE_FIELDS:
            attributes[field][batch_index, :length] = item.attribute_targets[field]
    return {
        "features": features,
        "frame_indices": frames,
        "event_targets": event_targets,
        "attribute_targets": attributes,
        "mask": mask,
        "items": items,
    }
