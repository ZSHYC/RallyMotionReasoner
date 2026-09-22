#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from tennisvar.generation.manifest import (
    MANIFEST_NAME,
    QWEN_ADAPTER_SCHEMA,
    load_qwen_adapter_manifest,
)
from tennisvar.generation.sft import prepare_qwen_sft_item, structured_video_messages
from tennisvar.io import read_json, read_jsonl, write_json
from tennisvar.schema_v2 import REQUIRED_SCHEMA_V2
from tennisvar.tracks import TRACK_TGTR_ASSISTED_PRED, normalize_track


def _distributed_indices(size: int, epoch: int, seed: int, rank: int, world_size: int) -> list[int]:
    if size <= 0:
        return []
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    total = math.ceil(size / world_size) * world_size
    indices = (indices * math.ceil(total / size))[:total]
    return indices[rank:total:world_size]


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*") if output_dir.is_dir() else []:
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if (path / "adapter_config.json").is_file() and (path / "trainer_state.pt").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _validate_data_report(
    report_path: Path,
    *,
    expected_track: str,
    train_data: Path,
    val_data: Path,
) -> dict[str, Any]:
    report = read_json(report_path)
    expected = {
        "schema": "tennisvar.qwen_data_report.v1",
        "status": "PASS",
        "track": expected_track,
        "media_mode": "videos",
        "gold_events_in_prompt": False,
        "candidate_fallback_used": False,
        "sampling_contract": {
            "max_frames": 32,
            "global_frames": 32,
            "local_frames": 4,
            "local_radius": 8,
            "top_k": 8,
            "candidate_frame_nms_radius": 8,
        },
    }
    mismatched = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
    if mismatched:
        raise ValueError(f"Qwen data-report contract mismatch: {mismatched}")
    for split, data in (("train", train_data), ("val", val_data)):
        row = (report.get("splits") or {}).get(split) or {}
        if (
            int(row.get("references") or -1) != int(row.get("sft_rows") or -2)
            or int(row.get("prompts") or -1) != int(row.get("sft_rows") or -2)
            or int(row.get("missing_media") or 0) != 0
            or int(row.get("leakage") or 0) != 0
            or int(row.get("empty_candidates") or 0) != 0
            or Path(str(row.get("sft_path") or "")).resolve() != data.resolve()
        ):
            raise ValueError(f"Qwen data-report {split} coverage/path contract mismatch")
    return report


def _validate_rows(
    rows: list[dict[str, Any]],
    label: str,
    expected_track: str,
    expected_split: str,
    *,
    validate_media: bool = True,
) -> set[str]:
    if not rows:
        raise ValueError(f"Qwen {label} split is empty")
    ids: set[str] = set()
    for row in rows:
        qa_id = str(row.get("qa_id") or "")
        if not qa_id or qa_id in ids:
            raise ValueError(f"Qwen {label} split has empty or duplicate qa_id: {qa_id}")
        ids.add(qa_id)
        structured_video_messages(row, include_answer=True)
        actual_track = normalize_track((row.get("e2e_metadata") or {}).get("track"))
        if actual_track != expected_track:
            raise ValueError(f"Qwen {label} row track mismatch for {qa_id}: {actual_track} != {expected_track}")
        metadata = row.get("e2e_metadata") or {}
        if metadata.get("split") != expected_split:
            raise ValueError(f"Qwen {label} row split mismatch for {qa_id}: {metadata.get('split')} != {expected_split}")
        if metadata.get("media_mode") != "videos" or list(metadata.get("schema_fields") or []) != list(REQUIRED_SCHEMA_V2):
            raise ValueError(f"Qwen {label} row media/schema contract mismatch for {qa_id}")
        if metadata.get("graph_source") != "region_fusion_predicted":
            raise ValueError(f"Qwen {label} row lacks predicted-event provenance: {qa_id}")
        if validate_media:
            missing = [str(path) for path in row["videos"][0] if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"Qwen {label} row has missing frames for {qa_id}: {missing[:3]}")
    return ids


def _validation_loss(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: Any,
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
    distributed: bool,
) -> float:
    import torch
    import torch.distributed as dist

    model.eval()
    total = torch.zeros(1, device=device)
    seen = torch.zeros(1, device=device)
    indices = _distributed_indices(len(rows), epoch, seed + 100_000, rank, world_size)
    with torch.no_grad():
        for row_index in indices:
            batch = prepare_qwen_sft_item(processor, rows[row_index], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch)
            total += output.loss.detach()
            seen += 1
    if distributed:
        dist.all_reduce(total)
        dist.all_reduce(seen)
    if not int(seen.item()):
        raise RuntimeError("native Qwen validation set produced no batches")
    return float((total / seen).cpu())


def _save_checkpoint(
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    *,
    rank: int,
    distributed: bool,
    global_step: int,
    next_epoch: int,
    next_iteration: int,
    manifest_base: dict[str, Any],
) -> Path:
    import torch
    import torch.distributed as dist

    local_rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
    }
    if distributed:
        rng_states: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(rng_states, local_rng)
        dist.barrier()
    else:
        rng_states = [local_rng]
    target = output_dir / f"checkpoint-{global_step}"
    if rank == 0 and not target.exists():
        temporary = output_dir / f".checkpoint-{global_step}.{os.getpid()}.tmp"
        if temporary.exists():
            raise FileExistsError(f"stale temporary checkpoint exists: {temporary}")
        staging_parent = Path(os.environ.get("TENNISVAR_LOCAL_CHECKPOINT_ROOT", "/tmp"))
        if not staging_parent.is_dir():
            raise FileNotFoundError(f"local Qwen checkpoint staging root does not exist: {staging_parent}")
        # Serialize on node-local storage before atomically publishing to the
        # run directory, which also supports network filesystems.
        with tempfile.TemporaryDirectory(
            prefix=f"tennisvar-qwen-checkpoint-{global_step}-", dir=staging_parent
        ) as staging_value:
            staging = Path(staging_value)
            base = model.module if hasattr(model, "module") else model
            base.save_pretrained(staging, safe_serialization=True)
            processor.save_pretrained(staging)
            torch.save(
                {
                    "schema": "tennisvar.qwen_native_trainer.v2",
                    "global_step": global_step,
                    "next_epoch": next_epoch,
                    "next_iteration": next_iteration,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_states": rng_states,
                    "world_size": int(manifest_base["world_size"]),
                    "seed": int(manifest_base["seed"]),
                    "track": str(manifest_base["track"]),
                    "training_contract": dict(manifest_base["training_contract"]),
                },
                staging / "trainer_state.pt",
            )
            manifest = {
                **manifest_base,
                "checkpoint_step": global_step,
                "checkpoint_storage_contract": "node_local_serialize_then_shared_atomic_publish",
            }
            (staging / MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shutil.copytree(staging, temporary)
            os.replace(temporary, target)
    if distributed:
        dist.barrier()
    if not target.is_dir():
        raise RuntimeError(f"checkpoint was not materialized: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Native multi-GPU Qwen3-VL LoRA SFT for TennisVAR.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-report", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for label, path in (("model", args.model), ("train", args.train_data), ("val", args.val_data)):
        if not path.exists():
            raise FileNotFoundError(f"missing Qwen {label}: {path}")
    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows:
        val_rows = val_rows[: args.max_val_rows]
    track = TRACK_TGTR_ASSISTED_PRED
    launch_rank = int(os.environ.get("RANK", "0"))
    data_report = (
        _validate_data_report(
            args.data_report,
            expected_track=track,
            train_data=args.train_data,
            val_data=args.val_data,
        )
        if args.data_report
        else None
    )
    # Validate media paths once on rank 0 to avoid redundant distributed I/O.
    train_ids = _validate_rows(
        train_rows,
        "train",
        track,
        "train",
        validate_media=launch_rank == 0,
    )
    val_ids = _validate_rows(
        val_rows,
        "val",
        track,
        "val",
        validate_media=launch_rank == 0,
    )
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Qwen train/val qa_id leakage: {sorted(overlap)[:10]}")
    dry_report = {
        "status": "DRY_RUN" if args.dry_run else "READY_TO_TRAIN",
        "schema": "tennisvar.qwen_native_training_report.v1",
        "track": track,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "data_report": str(args.data_report) if data_report else None,
        "distributed_backend": "torch_ddp",
        "gold_events_allowed": False,
    }
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("native Qwen LoRA training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl", device_id=device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    if distributed:
        dist.barrier()
    resume_checkpoint = _latest_checkpoint(args.output_dir) if args.resume else None
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(f"non-empty Qwen output directory has no resumable checkpoint: {args.output_dir}")

    if resume_checkpoint:
        load_qwen_adapter_manifest(
            resume_checkpoint,
            expected_track=track,
        )

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if resume_checkpoint:
        model = PeftModel.from_pretrained(base, str(resume_checkpoint), is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    for name, parameter in model.named_parameters():
        if "visual" in name and "lora_" in name:
            parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("native Qwen LoRA has no trainable parameters")
    training_model: Any = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False
    ) if distributed else model
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    per_rank_steps = math.ceil(len(train_rows) / world_size)
    optimizer_steps_per_epoch = math.ceil(per_rank_steps / args.gradient_accumulation_steps)
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    start_epoch, resume_iteration, global_step = 1, 0, 0
    if resume_checkpoint:
        state = torch.load(resume_checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
        if state.get("schema") != "tennisvar.qwen_native_trainer.v2":
            raise ValueError(f"invalid native Qwen trainer state: {resume_checkpoint}")
        expected_state = {
            "world_size": world_size,
            "seed": args.seed,
            "track": track,
            "training_contract": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "save_steps": args.save_steps,
            },
        }
        mismatched = {
            key: {"checkpoint": state.get(key), "current": value}
            for key, value in expected_state.items()
            if state.get(key) != value
        }
        if mismatched:
            raise ValueError(f"Qwen resume trainer contract mismatch: {mismatched}")
        rng_states = state.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != world_size or not isinstance(rng_states[rank], dict):
            raise ValueError("Qwen resume checkpoint has no per-rank RNG state")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["next_epoch"])
        resume_iteration = int(state["next_iteration"])
        global_step = int(state["global_step"])
        random.setstate(rng_states[rank]["python"])
        torch.set_rng_state(rng_states[rank]["torch"])
        torch.cuda.set_rng_state(rng_states[rank]["cuda"], device=device)

    manifest_base = {
        "schema": QWEN_ADAPTER_SCHEMA,
        "track": track,
        "dataset": "trace",
        "base_model": str(args.model),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "output_schema_fields": REQUIRED_SCHEMA_V2,
        "seed": args.seed,
        "trainer": "native_torch_ddp_peft",
        "world_size": world_size,
        "provenance_contract": "predicted_event+tgtr_checkpoint",
        "training_contract": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "save_steps": args.save_steps,
        },
    }
    started = time.time()
    history: list[dict[str, Any]] = []
    latest: Path | None = resume_checkpoint
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        indices = _distributed_indices(len(train_rows), epoch, args.seed, rank, world_size)
        training_model.train()
        loss_sum = torch.zeros(1, device=device)
        seen = torch.zeros(1, device=device)
        begin = resume_iteration if epoch == start_epoch else 0
        for iteration, row_index in enumerate(indices):
            if iteration < begin:
                continue
            boundary = (iteration + 1) % args.gradient_accumulation_steps == 0 or iteration + 1 == len(indices)
            synchronization = (
                training_model.no_sync() if distributed and not boundary else nullcontext()
            )
            group_start = (iteration // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(indices) - group_start)
            with synchronization:
                batch = prepare_qwen_sft_item(processor, train_rows[row_index], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = training_model(**batch)
                    loss = output.loss / group_size
                loss.backward()
            loss_sum += output.loss.detach()
            seen += 1
            if boundary:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    next_epoch = epoch + 1 if iteration + 1 == len(indices) else epoch
                    next_iteration = 0 if next_epoch != epoch else iteration + 1
                    latest = _save_checkpoint(
                        training_model, processor, optimizer, scheduler, args.output_dir, rank=rank,
                        distributed=distributed, global_step=global_step, next_epoch=next_epoch,
                        next_iteration=next_iteration, manifest_base=manifest_base,
                    )
        if distributed:
            dist.all_reduce(loss_sum)
            dist.all_reduce(seen)
        val_loss = _validation_loss(
            training_model,
            processor,
            val_rows,
            device,
            epoch=epoch,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )
        history.append(
            {"epoch": epoch, "train_loss": float((loss_sum / seen.clamp_min(1)).cpu()), "val_loss": val_loss}
        )
        latest = _save_checkpoint(
            training_model, processor, optimizer, scheduler, args.output_dir, rank=rank,
            distributed=distributed, global_step=global_step, next_epoch=epoch + 1,
            next_iteration=0, manifest_base=manifest_base,
        )
        resume_iteration = 0
    if rank == 0:
        report = {
            **dry_report,
            "status": "READY",
            "latest_checkpoint": str(latest),
            "output_dir": str(args.output_dir),
            "world_size": world_size,
            "global_step": global_step,
            "epochs_completed": max(0, args.epochs - start_epoch + 1),
            "history": history,
            "elapsed_seconds": time.time() - started,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "freeze_vit": True,
            "freeze_aligner": True,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
        write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
