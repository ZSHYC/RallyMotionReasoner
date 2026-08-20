from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tennisvar.data.trace import DATASET_NAME, load_source, sha256_file
from tennisvar.data_fingerprint import directory_sha256, file_sha256
from tennisvar.io import write_json, write_jsonl

from .checkpoint import load_event_checkpoint
from .dataset import event_feature_file, safe_name
from .decoder import PeakDecoder
from .graph import build_predicted_graph


def _decode_cached(model: Any, checkpoint: dict[str, Any], cache: dict[str, Any], device: Any, fps: float):
    import torch

    features = cache.get("features")
    frame_indices = cache.get("frame_indices")
    if features is None or frame_indices is None or not cache.get("provenance"):
        raise ValueError("invalid event feature cache: features, frame_indices and provenance are required")
    provenance = cache["provenance"]
    feature_contract = (checkpoint.get("run_manifest") or {}).get("feature_contract") or {}
    for field in ("backend", "weights_sha256", "tracknet_checkpoint_sha256"):
        if feature_contract.get(field) != provenance.get(field):
            raise ValueError(f"event feature/checkpoint provenance mismatch: {field}")
    expected = int(checkpoint["model_config"]["input_dim"])
    if features.ndim != 2 or int(features.shape[-1]) != expected:
        raise ValueError(f"event feature dimension mismatch: {tuple(features.shape)} expected (*, {expected})")
    frames = [int(value) for value in frame_indices.tolist()]
    with torch.no_grad():
        outputs = model(features.unsqueeze(0).to(device), torch.ones(1, len(frames), dtype=torch.bool, device=device))
    scores = torch.sigmoid(outputs["event_logits"][0]).detach().cpu().tolist()
    attributes = {
        field: torch.softmax(outputs[f"{field}_logits"][0], dim=-1).detach().cpu().tolist()
        for field in checkpoint["attribute_maps"]
        if f"{field}_logits" in outputs
    }
    decoder_cfg = checkpoint.get("decoder") or {}
    decoder = PeakDecoder(
        threshold=float(decoder_cfg.get("threshold", 0.5)),
        min_separation_frames=int(decoder_cfg.get("min_separation_frames", decoder_cfg.get("nms_frames", 8))),
        top_k=int(decoder_cfg.get("top_k", 32)),
    )
    return decoder.decode(scores, frames, attributes, checkpoint["attribute_maps"], fps=fps), scores, frames, features


def _shot_cache(
    events: list[Any], frames: list[int], features: Any, checkpoint: dict[str, Any], *,
    checkpoint_sha256: str, rally_id: str, split: str
) -> dict[str, Any]:
    import torch

    indices = [min(range(len(frames)), key=lambda index: abs(frames[index] - event.frame)) for event in events]
    selected = features[torch.tensor(indices, dtype=torch.long)] if indices else features.new_zeros((0, features.shape[-1]))
    return {
        "schema": "tennisvar.tgtr_event_features.v2",
        "source": "f3ed_predicted",
        "rally_id": rally_id,
        "split": split,
        "shot_ids": list(range(1, len(events) + 1)),
        "shot_frames": [event.frame for event in events],
        "shot_features": selected.float().cpu(),
        "event_checkpoint_data_fingerprint": checkpoint.get("data_fingerprint"),
        "event_checkpoint_sha256": checkpoint_sha256,
    }


def _valid_shot_cache(
    cache: dict[str, Any], *, rally_id: str, split: str, checkpoint: dict[str, Any], checkpoint_sha256: str
) -> bool:
    features = cache.get("shot_features")
    shot_ids = cache.get("shot_ids")
    frames = cache.get("shot_frames")
    return bool(
        cache.get("schema") == "tennisvar.tgtr_event_features.v2"
        and cache.get("source") == "f3ed_predicted"
        and cache.get("rally_id") == rally_id
        and cache.get("split") == split
        and cache.get("event_checkpoint_data_fingerprint") == checkpoint.get("data_fingerprint")
        and cache.get("event_checkpoint_sha256") == checkpoint_sha256
        and features is not None
        and getattr(features, "ndim", None) == 2
        and int(features.shape[0]) == len(shot_ids or []) == len(frames or [])
    )


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    import torch

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def predict_cached_split(
    *,
    checkpoint_path: Path,
    source_json: Path,
    feature_root: Path,
    frame_root: Path,
    output_root: Path,
    tgtr_feature_root: Path,
    split: str,
    dataset_name: str = DATASET_NAME,
    device: str | None = None,
    limit: int = 0,
    allow_empty_events: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Decode a complete split from immutable frame caches and export TGTR-ready artifacts."""
    import torch

    source_json = Path(source_json)
    rows = load_source(source_json)
    if limit > 0:
        rows = rows[:limit]
    graph_path = Path(output_root) / "graphs_f3ed" / f"sgtr_graph_f3ed_{dataset_name}_{split}.jsonl"
    prediction_path = Path(output_root) / "f3ed_predictions" / f"f3ed_{dataset_name}_{split}.jsonl"
    report_path = Path(output_root) / "manifests" / f"f3ed_export_{split}.json"
    for target in (graph_path, prediction_path, report_path):
        if target.exists() and not resume:
            raise FileExistsError(f"refusing to overwrite batch F3ED artifact: {target}")
    model, checkpoint = load_event_checkpoint(Path(checkpoint_path), device=device)
    checkpoint_sha256 = file_sha256(Path(checkpoint_path))
    actual_source_hash = sha256_file(source_json)
    expected_source_hash = (checkpoint.get("source_hashes") or {}).get(split)
    if expected_source_hash and expected_source_hash != actual_source_hash:
        raise ValueError(
            f"event checkpoint/source hash mismatch for {split}: {expected_source_hash} != {actual_source_hash}"
        )
    runtime_device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    empty: list[str] = []
    resumed_shot_features = 0
    started = time.time()
    for row in rows:
        rally_id = str(row["video"])
        cache_path = event_feature_file(Path(feature_root), split, rally_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing event feature cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        fps = float(row.get("fps") or 25.0)
        events, scores, frames, features = _decode_cached(model, checkpoint, cache, runtime_device, fps)
        if not events:
            empty.append(rally_id)
        prediction = {
            "schema": "tennisvar.f3ed.prediction.v2",
            "dataset": dataset_name,
            "split": split,
            "rally_id": rally_id,
            "mode": "predicted",
            "predicted_events": [event.to_json() for event in events],
            "frame_scores": scores,
            "frame_indices": frames,
            "feature_provenance": cache["provenance"],
            "checkpoint_data_fingerprint": checkpoint.get("data_fingerprint"),
            "checkpoint_sha256": checkpoint_sha256,
        }
        predictions.append(prediction)
        predicted_graph = build_predicted_graph(
                events,
                rally_id=rally_id,
                frames_dir=Path(frame_root) / rally_id,
                fps=fps,
                num_frames=int(row.get("num_frames") or len(frames)),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
                split=split,
            )
        predicted_graph["event_checkpoint_data_fingerprint"] = checkpoint.get("data_fingerprint")
        predicted_graph["event_checkpoint_sha256"] = checkpoint_sha256
        graphs.append(predicted_graph)
        target = Path(tgtr_feature_root) / split / f"{safe_name(rally_id)}.pt"
        if target.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite TGTR feature cache: {target}")
            existing = torch.load(target, map_location="cpu", weights_only=False)
            if not _valid_shot_cache(
                existing, rally_id=rally_id, split=split, checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
            ):
                raise RuntimeError(f"invalid or incompatible TGTR feature cache; refusing to resume: {target}")
            if list(existing.get("shot_frames") or []) != [event.frame for event in events]:
                raise RuntimeError(f"TGTR feature cache/event decode mismatch; refusing to resume: {target}")
            resumed_shot_features += 1
        else:
            _atomic_torch_save(
                _shot_cache(
                    events, frames, features, checkpoint, checkpoint_sha256=checkpoint_sha256,
                    rally_id=rally_id, split=split,
                ),
                target,
            )
    write_jsonl(prediction_path, predictions)
    write_jsonl(graph_path, graphs)
    status = "READY" if not empty else "DEGRADED_EMPTY_EVENTS"
    report = {
        "schema": "tennisvar.f3ed_export.v2",
        "status": status,
        "dataset": dataset_name,
        "split": split,
        "requested": len(rows),
        "predictions": len(predictions),
        "empty_event_rallies": len(empty),
        "empty_event_sample": empty[:50],
        "resumed_shot_features": resumed_shot_features,
        "gold_events_used_for_predictions": False,
        "prediction_path": str(prediction_path),
        "graph_path": str(graph_path),
        "tgtr_feature_root": str(tgtr_feature_root),
        "elapsed_seconds": time.time() - started,
        "checkpoint_data_fingerprint": checkpoint.get("data_fingerprint"),
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": actual_source_hash,
        "prediction_sha256": file_sha256(prediction_path),
        "graph_sha256": file_sha256(graph_path),
        "tgtr_feature_directory_sha256": directory_sha256(Path(tgtr_feature_root) / split),
    }
    write_json(report_path, report)
    if empty and not allow_empty_events:
        raise RuntimeError(
            f"{len(empty)} rallies produced zero predicted events; artifacts are marked DEGRADED and cannot be used as main baseline"
        )
    return report
