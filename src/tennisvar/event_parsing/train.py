from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tennisvar.data.trace import iter_event_labels, load_source, sha256_file
from tennisvar.data_fingerprint import directory_sha256
from tennisvar.evaluation.event_metrics import evaluate_event_records
from tennisvar.io import write_json

from .checkpoint import save_event_checkpoint
from .dataset import EventSequenceDataset, collate_event_sequences
from .decoder import PeakDecoder
from .labels import ATTRIBUTE_FIELDS, build_attribute_maps, decode_attribute
from .model import EventModelConfig, EventParsingModule, event_detector_loss


@dataclass(frozen=True)
class EventTrainingConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    target_radius: int = 4
    min_separation_frames: int = 8
    top_k: int = 32
    max_hours: float = 4.0
    seed: int = 42
    limit_train: int = 0
    limit_val: int = 0


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        **batch,
        "features": batch["features"].to(device),
        "event_targets": batch["event_targets"].to(device),
        "mask": batch["mask"].to(device),
        "attribute_targets": {key: value.to(device) for key, value in batch["attribute_targets"].items()},
    }


def _gold_from_item(item: Any, maps: dict[str, dict[str, int]]) -> tuple[list[int], list[dict[str, str | None]]]:
    indices = [index for index, value in enumerate(item.event_targets.tolist()) if float(value) >= 0.999]
    frames = [int(item.frame_indices[index]) for index in indices]
    attributes: list[dict[str, str | None]] = []
    for index in indices:
        attributes.append(
            {
                field: decode_attribute(field, int(item.attribute_targets[field][index]), maps)
                for field in ATTRIBUTE_FIELDS
            }
        )
    return frames, attributes


def _validation_outputs(model: Any, loader: Any, device: Any, maps: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            outputs = model(batch["features"], batch["mask"])
            for batch_index, item in enumerate(raw["items"]):
                length = int(item.features.shape[0])
                probabilities = torch.sigmoid(outputs["event_logits"][batch_index, :length]).cpu().tolist()
                attr_probabilities = {
                    field: torch.softmax(outputs[f"{field}_logits"][batch_index, :length], dim=-1).cpu().tolist()
                    for field in maps
                    if f"{field}_logits" in outputs
                }
                gold_frames, gold_attributes = _gold_from_item(item, maps)
                rows.append(
                    {
                        "rally_id": item.rally_id,
                        "frames": [int(value) for value in item.frame_indices.tolist()],
                        "scores": probabilities,
                        "attributes": attr_probabilities,
                        "gold_frames": gold_frames,
                        "gold_attributes": gold_attributes,
                    }
                )
    return rows


def _validation_loss(model: Any, loader: Any, device: Any) -> float:
    import torch

    total = 0.0
    seen = 0
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            outputs = model(batch["features"], batch["mask"])
            losses = event_detector_loss(outputs, batch["event_targets"], batch["attribute_targets"], batch["mask"])
            count = len(raw["items"])
            total += float(losses["loss"].detach().cpu()) * count
            seen += count
    if not seen:
        raise RuntimeError("validation loader is empty")
    return total / seen


def _decode_validation(rows: list[dict[str, Any]], maps: dict[str, dict[str, int]], decoder: PeakDecoder) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        predicted = decoder.decode(row["scores"], row["frames"], row["attributes"], maps, fps=25.0)
        gold_set = set(row["gold_frames"])
        records.append(
            {
                "rally_id": row["rally_id"],
                "predicted_frames": [event.frame for event in predicted],
                "predicted_attributes": [event.attributes for event in predicted],
                "gold_frames": row["gold_frames"],
                "gold_attributes": row["gold_attributes"],
                "frame_scores": row["scores"],
                "frame_labels": [int(frame in gold_set) for frame in row["frames"]],
            }
        )
    return records


def train_event_detector(
    source_root: Path,
    feature_root: Path,
    run_dir: Path,
    config: EventTrainingConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    source_root, feature_root, run_dir = Path(source_root), Path(feature_root), Path(run_dir)
    trainer_state_path = run_dir / "trainer_state.pt"
    final_checkpoint_path = run_dir / "event_detector.pt"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        if final_checkpoint_path.exists():
            raise FileExistsError(f"event run is already complete: {final_checkpoint_path}")
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"event run has no resumable trainer state: {trainer_state_path}")
    else:
        if resume:
            raise FileNotFoundError(f"event run does not exist for --resume: {run_dir}")
        run_dir.mkdir(parents=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    train_source, val_source = source_root / "train.json", source_root / "val.json"
    source_hashes = {split: sha256_file(source_root / f"{split}.json") for split in ("train", "val", "test")}
    feature_hashes = {split: directory_sha256(feature_root / split) for split in ("train", "val", "test")}
    train_rows = load_source(train_source)
    attribute_maps = build_attribute_maps(iter_event_labels(train_rows))
    train_data = EventSequenceDataset(
        train_source,
        feature_root,
        "train",
        attribute_maps,
        target_radius=config.target_radius,
        limit=config.limit_train,
    )
    val_data = EventSequenceDataset(
        val_source,
        feature_root,
        "val",
        attribute_maps,
        target_radius=config.target_radius,
        limit=config.limit_val,
    )
    if not train_data or not val_data:
        raise RuntimeError("event detector requires non-empty train and validation data")
    first_provenance = train_data[0].provenance
    feature_contract = {
        "backend": first_provenance.get("backend"),
        "weights_sha256": first_provenance.get("weights_sha256"),
        "tracknet_checkpoint_sha256": first_provenance.get("tracknet_checkpoint_sha256"),
        "feature_dim": first_provenance.get("feature_dim"),
    }
    for item in [*train_data.items, *val_data.items]:
        contract = {
            "backend": item.provenance.get("backend"),
            "weights_sha256": item.provenance.get("weights_sha256"),
            "tracknet_checkpoint_sha256": item.provenance.get("tracknet_checkpoint_sha256"),
            "feature_dim": item.provenance.get("feature_dim"),
        }
        if contract != feature_contract:
            raise ValueError(f"mixed event feature provenance is forbidden: {item.rally_id}")
    input_dim = int(train_data[0].features.shape[-1])
    model_config = EventModelConfig(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventParsingModule(model_config, attribute_maps).to(device)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    training_model = torch.nn.DataParallel(model) if gpu_count > 1 else model
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_event_sequences,
    )
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_event_sequences)
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    started = time.time()
    deadline = started + config.max_hours * 3600
    elapsed_before = 0.0
    stopped_due_time = False
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    start_epoch = 1
    if resume:
        state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        if state.get("schema") != "tennisvar.f3ed_trainer.v1":
            raise ValueError(f"invalid event trainer state schema: {state.get('schema')}")
        if state.get("config") != asdict(config):
            raise ValueError("event trainer state config does not match requested config")
        if state.get("source_hashes") != source_hashes:
            raise ValueError("event trainer state source hashes do not match immutable data")
        if state.get("feature_hashes") != feature_hashes:
            raise ValueError("event trainer state feature hashes do not match immutable caches")
        if state.get("attribute_maps") != attribute_maps or state.get("model_config") != model_config.to_json():
            raise ValueError("event trainer state model/label contract mismatch")
        if state.get("feature_contract") != feature_contract or int(state.get("gpu_count", -1)) != gpu_count:
            raise ValueError("event trainer state feature/GPU contract mismatch")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["generator_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        history = list(state.get("history") or [])
        best_loss = float(state["best_loss"])
        best_state = state.get("best_state")
        start_epoch = int(state["last_completed_epoch"]) + 1
        elapsed_before = float(state.get("elapsed_seconds") or 0.0)

    for epoch in range(start_epoch, config.epochs + 1):
        if time.time() >= deadline:
            stopped_due_time = True
            break
        training_model.train()
        total_loss = 0.0
        seen = 0
        epoch_complete = True
        for raw in train_loader:
            if time.time() >= deadline:
                stopped_due_time = True
                epoch_complete = False
                break
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = training_model(batch["features"], batch["mask"])
            losses = event_detector_loss(outputs, batch["event_targets"], batch["attribute_targets"], batch["mask"])
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(training_model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(losses["loss"].detach().cpu()) * len(raw["items"])
            seen += len(raw["items"])
        if not epoch_complete:
            break
        if not seen:
            break
        value = total_loss / seen
        val_loss = _validation_loss(training_model, val_loader, device)
        history.append({"epoch": float(epoch), "train_loss": value, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        trainer_state = {
            "schema": "tennisvar.f3ed_trainer.v1",
            "last_completed_epoch": epoch,
            "model_state": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
            "best_state": best_state,
            "best_loss": best_loss,
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "history": history,
            "config": asdict(config),
            "model_config": model_config.to_json(),
            "attribute_maps": attribute_maps,
            "source_hashes": source_hashes,
            "feature_hashes": feature_hashes,
            "feature_contract": feature_contract,
            "gpu_count": gpu_count,
            "elapsed_seconds": elapsed_before + time.time() - started,
        }
        temporary_state = trainer_state_path.with_name(f".{trainer_state_path.name}.{os.getpid()}.tmp")
        torch.save(trainer_state, temporary_state)
        os.replace(temporary_state, trainer_state_path)
    if len(history) < config.epochs:
        report = {
            "status": "PARTIAL_TIME_LIMIT" if stopped_due_time else "NO_GO_INCOMPLETE",
            "epochs_completed": len(history),
            "epochs_requested": config.epochs,
            "resumable_state": str(trainer_state_path) if trainer_state_path.is_file() else None,
            "elapsed_seconds": elapsed_before + time.time() - started,
        }
        write_json(run_dir / "metrics.json", report)
        return report
    if best_state is None:
        raise RuntimeError("event detector training completed no epoch")
    model.load_state_dict(best_state)
    model.to(device).eval()
    raw_validation = _validation_outputs(model, val_loader, device, attribute_maps)
    candidates = [round(0.05 * value, 2) for value in range(1, 20)]
    scored: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        decoder = PeakDecoder(threshold, config.min_separation_frames, config.top_k)
        report = evaluate_event_records(_decode_validation(raw_validation, attribute_maps, decoder))
        scored.append((threshold, report))
    threshold, validation = max(scored, key=lambda item: item[1]["overall"].get("event_f1@8", 0.0))
    run_manifest = {
        "schema": "tennisvar.run.v1",
        "run_id": run_dir.name,
        "seed": config.seed,
        "device": str(device),
        "gpu_count": gpu_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix": started,
        "elapsed_seconds": elapsed_before + time.time() - started,
        "config": asdict(config),
        "config_sha256": hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
        "source_hashes": source_hashes,
        "feature_hashes": feature_hashes,
        "feature_contract": feature_contract,
        "stopped_due_time_limit": stopped_due_time,
        "event_feature_backend": str(train_data[0].provenance.get("backend") or ""),
        "tracknet_checkpoint_sha256": train_data[0].provenance.get("tracknet_checkpoint_sha256"),
    }
    decoder_config = {"threshold": threshold, "min_separation_frames": config.min_separation_frames, "top_k": config.top_k}
    checkpoint = final_checkpoint_path
    save_event_checkpoint(
        checkpoint,
        model,
        attribute_maps=attribute_maps,
        source_hashes=source_hashes,
        feature_hashes=feature_hashes,
        decoder=decoder_config,
        metrics=validation,
        run_manifest=run_manifest,
    )
    report = {
        "status": "READY" if validation["overall"].get("event_f1@8", 0.0) > 0 else "NO_GO_ZERO_EVENT_F1",
        "checkpoint": str(checkpoint),
        "train_sequences": len(train_data),
        "val_sequences": len(val_data),
        "best_val_loss": best_loss,
        "epochs_completed": len(history),
        "decoder": decoder_config,
        "validation": validation,
        "run_manifest": run_manifest,
        "history": history,
    }
    write_json(run_dir / "metrics.json", report)
    return report
