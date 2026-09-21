#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tennisvar.config import DEFAULT_CONFIG
from tennisvar.configs import (
    dump_simple_yaml,
    load_tgtr_vl_config,
    module_summary,
    resolve_paths_config,
    tgtr_vl_feature_root,
)
from tennisvar.data.graph_qa import (
    collate,
    dataset_relative_root,
    graph_file,
    make_graph_datasets,
    move_batch,
    refs_file,
)
from tennisvar.data_fingerprint import directory_sha256, file_sha256
from tennisvar.io import write_json
from tennisvar.tactical_reasoning.model import TacticalGraphGuidedTemporalReasoner
from tennisvar.training.eval import evaluate_internal
from tennisvar.training.losses import compute_loss


def build_model(cfg: dict[str, Any], vocab_size: int, label_maps: dict[str, dict[str, int]]) -> TacticalGraphGuidedTemporalReasoner:
    model_cfg = cfg.get("model", {})
    feature_cfg = cfg.get("feature_extraction", {})
    motion_dim = int(feature_cfg.get("motion_feature_dim", 0))
    if str(cfg.get("module_flags", {}).get("motion_token_mode", "none")) == "none":
        motion_dim = 0
    return TacticalGraphGuidedTemporalReasoner(
        vocab_size,
        label_maps,
        visual_feature_dim=int(feature_cfg.get("feature_dim", 800)),
        motion_feature_dim=motion_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train dataset-neutral TennisVAR-TGTR on predicted event graphs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/tennisvar.yaml"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=4.0)
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"training.batch_size={args.batch_size}")
    if args.lr is not None:
        overrides.append(f"training.lr={args.lr}")
    if args.name:
        overrides.append(f"experiment_name={args.name}")

    cfg = load_tgtr_vl_config(args.experiment_config, overrides)
    paths = resolve_paths_config(args.config)
    feature_root = tgtr_vl_feature_root(paths, cfg)
    training_cfg = cfg.get("training", {})
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    default_run = f"{cfg.get('experiment_name', 'tgtr_vl')}_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    output_dir = args.output_dir or paths["runs_root"] / default_run
    train_ds, val_ds, meta = make_graph_datasets(paths, cfg, feature_root)
    source = str(meta["graph_source"])
    data_files = {
        "train_refs": refs_file(paths, "train", cfg),
        "val_refs": refs_file(paths, "val", cfg),
        "train_graphs": graph_file(paths, "train", source, cfg),
        "val_graphs": graph_file(paths, "val", source, cfg),
        "train_feature_export": dataset_relative_root(paths, cfg, "event_export_root_name", "manifests")
        / "f3ed_export_train.json",
        "val_feature_export": dataset_relative_root(paths, cfg, "event_export_root_name", "manifests")
        / "f3ed_export_val.json",
        "test_feature_export": dataset_relative_root(paths, cfg, "event_export_root_name", "manifests")
        / "f3ed_export_test.json",
    }
    data_hashes = {key: file_sha256(path) for key, path in data_files.items()}
    for split in ("train", "val", "test"):
        data_hashes[f"{split}_features"] = directory_sha256(feature_root / split)
    export_reports = [
        json.loads(data_files[f"{split}_feature_export"].read_text(encoding="utf-8"))
        for split in ("train", "val", "test")
    ]
    upstream_event_hashes = {row.get("checkpoint_sha256") for row in export_reports}
    upstream_event_fingerprints = {row.get("checkpoint_data_fingerprint") for row in export_reports}
    if len(upstream_event_hashes) != 1 or len(upstream_event_fingerprints) != 1:
        raise ValueError("TGTR training inputs mix F3ED checkpoints")
    upstream_event_checkpoint_sha256 = next(iter(upstream_event_hashes))
    upstream_event_data_fingerprint = next(iter(upstream_event_fingerprints))
    for label, value in (
        ("checkpoint_sha256", upstream_event_checkpoint_sha256),
        ("data_fingerprint", upstream_event_data_fingerprint),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"TGTR training has invalid upstream F3ED {label}")
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    data_fingerprint = hashlib.sha256(json.dumps(data_hashes, sort_keys=True).encode()).hexdigest()
    report = {
        "status": "DRY_RUN" if args.dry_run else "READY_TO_RUN",
        "architecture": cfg.get("architecture", "TennisVAR-TGTR-v2"),
        "experiment_name": cfg.get("experiment_name"),
        "seed": seed,
        "output_dir": str(output_dir),
        "feature_root": str(feature_root),
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "missing_train_graphs": train_ds.missing_graphs,
        "missing_val_graphs": val_ds.missing_graphs,
        "vocab_size": len(meta["vocab"]),
        "label_sizes": {key: len(value) for key, value in meta["label_maps"].items()},
        "module_summary": module_summary(cfg),
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    trainer_state_path = output_dir / "trainer_state.pt"
    final_checkpoint_path = output_dir / "checkpoint.pt"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite TGTR run: {output_dir}")
        if final_checkpoint_path.exists():
            raise FileExistsError(f"TGTR run is already complete: {final_checkpoint_path}")
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"TGTR run has no resumable trainer state: {trainer_state_path}")
    else:
        if args.resume:
            raise FileNotFoundError(f"TGTR run does not exist for --resume: {output_dir}")
        output_dir.mkdir(parents=True)
        (output_dir / "resolved_config.yaml").write_text(dump_simple_yaml(cfg), encoding="utf-8")
        write_json(output_dir / "resolved_config.json", cfg)
        write_json(output_dir / "module_summary.json", module_summary(cfg))
        write_json(
            output_dir / "model_card.json",
            {
                "architecture": cfg.get("architecture", "TennisVAR-TGTR-v2"),
                "purpose": f"{meta['dataset']} predicted-event tennis video-graph-language reasoning",
                "visual_backbone": cfg.get("feature_extraction", {}).get("feature_set", "frozen_stats_v1"),
                "qwen_lora_enabled": bool(cfg.get("module_flags", {}).get("use_qwen_lora", False)),
                "claim_boundary": "TGTR is an evidence selector; the structured-prompt baseline is not claimed as joint end-to-end fusion.",
                "module_flags": module_summary(cfg),
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_contract = {
        "device_type": device.type,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_ds, batch_size=int(training_cfg.get("batch_size", 64)), shuffle=False, collate_fn=collate)
    model = build_model(cfg, len(meta["vocab"]), meta["label_maps"]).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("lr", 1.5e-3)), weight_decay=float(training_cfg.get("weight_decay", 1e-4)))
    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -math.inf
    best_state = None
    history = []
    started = time.time()
    elapsed_before = 0.0
    epochs = int(training_cfg.get("epochs", 120))
    loss_weights = {key: float(value) for key, value in cfg.get("loss_weights", {}).items()}
    start_epoch = 1
    if args.resume:
        state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        if state.get("schema") != "tennisvar.tgtr_trainer.v1":
            raise ValueError(f"invalid TGTR trainer state schema: {state.get('schema')}")
        if state.get("config_hash") != config_hash or state.get("data_fingerprint") != data_fingerprint:
            raise ValueError("TGTR trainer state config/data fingerprint mismatch")
        if state.get("runtime_contract") != runtime_contract:
            raise ValueError("TGTR trainer state GPU runtime contract mismatch")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        generator.set_state(state["generator_state"])
        random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        best_score = float(state["best_score"])
        best_state = state.get("best_state")
        history = list(state.get("history") or [])
        start_epoch = int(state["last_completed_epoch"]) + 1
        elapsed_before = float(state.get("elapsed_seconds") or 0.0)

    stopped_due_time = False
    for epoch in range(start_epoch, epochs + 1):
        if time.time() - started > args.max_hours * 3600:
            stopped_due_time = True
            break
        model.train()
        train_loss = 0.0
        seen = 0
        epoch_complete = True
        for raw_batch in train_loader:
            if time.time() - started > args.max_hours * 3600:
                stopped_due_time = True
                epoch_complete = False
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch)
                loss = compute_loss(outputs, batch, loss_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(batch["items"])
            seen += len(batch["items"])
        if not epoch_complete:
            break
        metrics = evaluate_internal(model, val_loader, device, loss_weights=loss_weights)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / seen if seen else 0.0
        history.append(metrics)
        score = metrics["evidence_f1"] + metrics["key_action_accuracy"] + metrics["level_3_accuracy"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        trainer_state = {
            "schema": "tennisvar.tgtr_trainer.v1",
            "last_completed_epoch": epoch,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "best_state": best_state,
            "best_score": best_score,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "generator_state": generator.get_state(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "history": history,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "runtime_contract": runtime_contract,
            "elapsed_seconds": elapsed_before + time.time() - started,
        }
        temporary_state = trainer_state_path.with_name(f".{trainer_state_path.name}.{os.getpid()}.tmp")
        torch.save(trainer_state, temporary_state)
        os.replace(temporary_state, trainer_state_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))

    if len(history) < epochs:
        partial = {
            **report,
            "status": "PARTIAL_TIME_LIMIT" if stopped_due_time else "NO_GO_INCOMPLETE",
            "epochs_completed": len(history),
            "epochs_requested": epochs,
            "resumable_state": str(trainer_state_path) if trainer_state_path.is_file() else None,
            "elapsed_sec": elapsed_before + time.time() - started,
        }
        write_json(output_dir / "metrics.json", partial)
        print(json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    if best_state is None:
        raise RuntimeError("TGTR training completed without a valid best checkpoint")
    model.load_state_dict(best_state)
    best_t = 0.45
    best_f1 = -1.0
    for threshold in training_cfg.get("evidence_threshold_grid", [0.45]):
        metrics = evaluate_internal(model, val_loader, device, evidence_threshold=float(threshold), loss_weights=loss_weights)
        if metrics["evidence_f1"] > best_f1:
            best_f1 = metrics["evidence_f1"]
            best_t = float(threshold)
    final_val = evaluate_internal(model, val_loader, device, evidence_threshold=best_t, loss_weights=loss_weights)
    checkpoint = final_checkpoint_path
    run_manifest = {
        "schema": "tennisvar.run.v1",
        "command": [sys.executable, *sys.argv],
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config_hash": config_hash,
        "data_hashes": data_hashes,
        "data_fingerprint": data_fingerprint,
        "upstream_event_checkpoint_sha256": upstream_event_checkpoint_sha256,
        "upstream_event_data_fingerprint": upstream_event_data_fingerprint,
        "elapsed_seconds": elapsed_before + time.time() - started,
    }
    checkpoint_payload = (
        {
            "schema": "tennisvar.tgtr.v1",
            "model_state": model.state_dict(),
            "vocab": meta["vocab"],
            "label_maps": meta["label_maps"],
            "hidden_dim": int(cfg.get("model", {}).get("hidden_dim", 256)),
            "num_layers": int(cfg.get("model", {}).get("num_layers", 2)),
            "num_heads": int(cfg.get("model", {}).get("num_heads", 8)),
            "dropout": float(cfg.get("model", {}).get("dropout", 0.1)),
            "visual_feature_dim": int(cfg.get("feature_extraction", {}).get("feature_dim", 800)),
            "motion_feature_dim": int(model.motion_feature_dim),
            "thresholds": {"evidence_threshold": best_t},
            "config": cfg,
            "architecture": cfg.get("architecture", "TennisVAR-TGTR-v2"),
            "graph_source": cfg.get("data", {}).get("graph_source", "f3ed"),
            "feature_root": str(feature_root),
            "seed": seed,
            "config_hash": config_hash,
            "data_hashes": data_hashes,
            "data_fingerprint": data_fingerprint,
            "upstream_event_checkpoint_sha256": upstream_event_checkpoint_sha256,
            "upstream_event_data_fingerprint": upstream_event_data_fingerprint,
            "run_manifest": run_manifest,
        }
    )
    temporary_checkpoint = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
    torch.save(checkpoint_payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint)
    report.update(
        {
            "status": "READY",
            "device": str(device),
            "epochs": epochs,
            "epochs_completed": len(history),
            "elapsed_sec": round(elapsed_before + time.time() - started, 4),
            "checkpoint": str(checkpoint),
            "thresholds": {"evidence_threshold": best_t},
            "best_val": final_val,
            "history_tail": history[-5:],
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        }
    )
    write_json(output_dir / "metrics.json", report)
    write_json(output_dir / "train_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
