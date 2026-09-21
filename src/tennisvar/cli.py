from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tennisvar.config import DEFAULT_CONFIG, load_paths
from tennisvar.io import write_json


def _path(args: argparse.Namespace, paths: dict[str, Path], name: str) -> Path:
    value = getattr(args, name, None) or paths.get(name)
    if value is None:
        raise ValueError(f"missing --{name.replace('_', '-')} and no value is configured")
    return Path(value)


def _emit(payload: dict[str, Any], output: Path | None) -> int:
    if output:
        write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def prepare_data(args: argparse.Namespace) -> int:
    from tennisvar.data.trace import DATASET_NAME, prepare_dataset

    paths = load_paths(args.config)
    report = prepare_dataset(
        args.source_root or paths["source_data_root"],
        args.output_root or paths["artifacts_root"] / DATASET_NAME,
        args.frame_root or paths["frame_root"],
        strict_media=not args.allow_missing_media,
    )
    return _emit(report, args.output)


def cache_features(args: argparse.Namespace) -> int:
    import torch

    from tennisvar.data.trace import DATASET_NAME, load_source
    from tennisvar.event_parsing.dataset import event_feature_file
    from tennisvar.event_parsing.features import DinoMotionFeatureExtractor
    from tennisvar.features import load_ball_track
    from tennisvar.video import list_images

    paths = load_paths(args.config)
    source_root = args.source_root or paths["source_data_root"]
    frame_root = args.frame_root or paths["frame_root"]
    output_root = args.output_root or paths["artifacts_root"] / DATASET_NAME / "event_features"
    extractor = DinoMotionFeatureExtractor(
        _path(args, paths, "dinov3_repo"),
        _path(args, paths, "dinov3_weights"),
        device=args.device,
        batch_size=args.batch_size,
        require_ball=args.require_ball,
    )
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    report: dict[str, Any] = {"schema": "tennisvar.epm_feature_export.v1", "splits": {}}
    for split in splits:
        rows = load_source(Path(source_root) / f"{split}.json")
        if args.limit:
            rows = rows[: args.limit]
        written = skipped = 0
        for row in rows:
            rally_id = str(row["video"])
            target = event_feature_file(output_root, split, rally_id)
            if target.exists() and args.resume:
                skipped += 1
                continue
            if target.exists():
                raise FileExistsError(f"refusing to overwrite feature cache: {target}")
            frames = list_images(Path(frame_root) / rally_id)
            if not frames:
                raise FileNotFoundError(f"missing frames for {split}/{rally_id}")
            track = load_ball_track(args.ball_track_root or paths.get("ball_track_root"), split, rally_id)
            features, provenance = extractor.extract(frames, frame_indices=list(range(len(frames))), ball_track=track)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            torch.save(
                {
                    "schema": "tennisvar.event_features.v1",
                    "rally_id": rally_id,
                    "split": split,
                    "features": features,
                    "frame_indices": torch.arange(len(frames)),
                    "provenance": provenance.to_json(),
                },
                temporary,
            )
            os.replace(temporary, target)
            written += 1
        report["splits"][split] = {"requested": len(rows), "written": written, "skipped": skipped}
    report["status"] = "READY"
    report["output_root"] = str(output_root)
    return _emit(report, args.output)


def train_epm(args: argparse.Namespace) -> int:
    from tennisvar.data.trace import DATASET_NAME
    from tennisvar.event_parsing.train import EventTrainingConfig, train_event_detector

    paths = load_paths(args.config)
    config = EventTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_hours=args.max_hours,
        seed=args.seed,
    )
    report = train_event_detector(
        args.source_root or paths["source_data_root"],
        args.feature_root or paths["artifacts_root"] / DATASET_NAME / "event_features",
        args.output_dir,
        config,
        resume=args.resume,
    )
    return _emit(report, args.output)


def export_events(args: argparse.Namespace) -> int:
    from tennisvar.data.trace import DATASET_NAME
    from tennisvar.event_parsing.batch import predict_cached_split

    paths = load_paths(args.config)
    artifact_root = args.output_root or paths["artifacts_root"] / DATASET_NAME
    report = predict_cached_split(
        checkpoint_path=args.checkpoint,
        source_json=(args.source_root or paths["source_data_root"]) / f"{args.split}.json",
        feature_root=args.feature_root or paths["artifacts_root"] / DATASET_NAME / "event_features",
        frame_root=args.frame_root or paths["frame_root"],
        output_root=artifact_root,
        tgtr_feature_root=artifact_root / "tgtr_event_features" / "f3ed_event_features",
        split=args.split,
        dataset_name=DATASET_NAME,
        device=args.device,
        resume=args.resume,
    )
    return _emit(report, args.output)


def predict_events(args: argparse.Namespace) -> int:
    from tennisvar.event_parsing.region_features import load_track_payload
    from tennisvar.event_parsing.runtime import build_event_predictor
    from tennisvar.media import materialize_video

    paths = load_paths(args.config)
    predictor = build_event_predictor(
        args.checkpoint,
        dinov3_repo=_path(args, paths, "dinov3_repo"),
        dinov3_weights=_path(args, paths, "dinov3_weights"),
        device=args.device,
    )
    with materialize_video(args.video, fps=args.fps) as media:
        result = predictor.predict(
            media.frame_paths,
            fps=media.fps,
            ball_track=load_track_payload(args.ball_track) if args.ball_track else None,
        )
        payload = {
            "schema": "tennisvar.epm.prediction.v1",
            "rally_id": Path(args.video).stem,
            "fps": media.fps,
            "events": [event.to_json() for event in result.events],
            "feature_provenance": result.feature_provenance.to_json(),
            "checkpoint_provenance": result.checkpoint_provenance,
        }
    return _emit(payload, args.output)


def predict(args: argparse.Namespace) -> int:
    from tennisvar.pipeline import TennisVAR

    paths = load_paths(args.config)
    model = TennisVAR(
        event_checkpoint=args.event_checkpoint,
        tgtr_checkpoint=args.tgtr_checkpoint,
        qwen_model=args.qwen_model or paths["qwen3_vl_model"],
        qwen_adapter=args.qwen_adapter,
        dinov3_repo=_path(args, paths, "dinov3_repo"),
        dinov3_weights=_path(args, paths, "dinov3_weights"),
        device=args.device,
    )
    payload = model.predict(args.video, args.question, fps=args.fps, ball_track=args.ball_track)
    return _emit(payload, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tennisvar", description="Stroke-evidence-grounded tennis video reasoning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="validate TRACE splits and build graph/QA artifacts")
    prepare.add_argument("--source-root", type=Path)
    prepare.add_argument("--output-root", type=Path)
    prepare.add_argument("--frame-root", type=Path)
    prepare.add_argument("--allow-missing-media", action="store_true")
    prepare.add_argument("--output", type=Path)
    prepare.set_defaults(handler=prepare_data)

    cache = commands.add_parser("cache-epm-features", help="extract DINOv3, motion and TrackNet frame features")
    cache.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    cache.add_argument("--source-root", type=Path)
    cache.add_argument("--frame-root", type=Path)
    cache.add_argument("--output-root", type=Path)
    cache.add_argument("--dinov3-repo", type=Path)
    cache.add_argument("--dinov3-weights", type=Path)
    cache.add_argument("--ball-track-root", type=Path)
    cache.add_argument("--batch-size", type=int, default=32)
    cache.add_argument("--device")
    cache.add_argument("--limit", type=int, default=0)
    cache.add_argument("--require-ball", action="store_true")
    cache.add_argument("--resume", action="store_true")
    cache.add_argument("--output", type=Path)
    cache.set_defaults(handler=cache_features)

    train = commands.add_parser("train-epm", help="train the Event Parsing Module")
    train.add_argument("--source-root", type=Path)
    train.add_argument("--feature-root", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--max-hours", type=float, default=48.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--output", type=Path)
    train.set_defaults(handler=train_epm)

    export = commands.add_parser("export-events", help="export predicted event graphs and TGTR event features")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--split", choices=["train", "val", "test"], required=True)
    export.add_argument("--source-root", type=Path)
    export.add_argument("--frame-root", type=Path)
    export.add_argument("--feature-root", type=Path)
    export.add_argument("--output-root", type=Path)
    export.add_argument("--device")
    export.add_argument("--resume", action="store_true")
    export.add_argument("--output", type=Path)
    export.set_defaults(handler=export_events)

    epm = commands.add_parser("predict-events", help="run region-motion event detection or a legacy EPM file")
    epm.add_argument("--video", type=Path, required=True)
    epm.add_argument("--checkpoint", type=Path, required=True, help="region expert directory or legacy EPM checkpoint")
    epm.add_argument("--dinov3-repo", type=Path)
    epm.add_argument("--dinov3-weights", type=Path)
    epm.add_argument("--ball-track", type=Path, required=True, help="TrackNet JSON payload or external CSV")
    epm.add_argument("--device")
    epm.add_argument("--fps", type=float)
    epm.add_argument("--output", type=Path)
    epm.set_defaults(handler=predict_events)

    full = commands.add_parser("predict", help="run EPM -> TGTR -> grounded Qwen3-VL generation")
    full.add_argument("--video", type=Path, required=True)
    full.add_argument("--question", required=True)
    full.add_argument("--event-checkpoint", type=Path, required=True, help="region expert directory or legacy EPM checkpoint")
    full.add_argument("--tgtr-checkpoint", type=Path, required=True)
    full.add_argument("--qwen-model", type=Path)
    full.add_argument("--qwen-adapter", type=Path)
    full.add_argument("--dinov3-repo", type=Path)
    full.add_argument("--dinov3-weights", type=Path)
    full.add_argument("--ball-track", type=Path, required=True, help="TrackNet JSON payload or external CSV")
    full.add_argument("--device")
    full.add_argument("--fps", type=float)
    full.add_argument("--output", type=Path)
    full.set_defaults(handler=predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
