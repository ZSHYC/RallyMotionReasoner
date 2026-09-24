from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rallymotionreasoner.config import DEFAULT_CONFIG, load_paths
from rallymotionreasoner.io import write_json


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
    from rallymotionreasoner.data.trace import DATASET_NAME, prepare_dataset

    paths = load_paths(args.config)
    report = prepare_dataset(
        args.source_root or paths["source_data_root"],
        args.output_root or paths["artifacts_root"] / DATASET_NAME,
        args.frame_root or paths["frame_root"],
        strict_media=not args.allow_missing_media,
    )
    return _emit(report, args.output)


def predict_events(args: argparse.Namespace) -> int:
    from rallymotionreasoner.event_detection.runtime import EventPredictor
    from rallymotionreasoner.features.ball_trajectory import load_track_payload
    from rallymotionreasoner.media import materialize_video

    paths = load_paths(args.config)
    predictor = EventPredictor(
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
            video_size=media.source_size,
        )
        payload = {
            "schema": "rallymotionreasoner.event.prediction.v1",
            "rally_id": Path(args.video).stem,
            "fps": media.fps,
            "events": [event.to_json() for event in result.events],
            "feature_provenance": result.feature_provenance.to_json(),
            "checkpoint_provenance": result.checkpoint_provenance,
        }
    return _emit(payload, args.output)


def export_events(args: argparse.Namespace) -> int:
    import torch

    from rallymotionreasoner.configs import graph_reasoner_feature_root, load_graph_reasoner_config
    from rallymotionreasoner.data.graph_qa import graph_file, safe_feature_name
    from rallymotionreasoner.data.trace import load_source
    from rallymotionreasoner.event_detection.features import shot_feature_payload
    from rallymotionreasoner.event_detection.graph import build_predicted_graph
    from rallymotionreasoner.event_detection.runtime import EventPredictor
    from rallymotionreasoner.features.ball_trajectory import load_track_payload
    from rallymotionreasoner.io import write_jsonl
    from rallymotionreasoner.video import list_images

    paths = load_paths(args.config)
    cfg = load_graph_reasoner_config(args.experiment_config)
    rows = load_source(paths["source_data_root"] / f"{args.split}.json")
    graph_path = graph_file(paths, args.split, "motion_region", cfg)
    feature_root = graph_reasoner_feature_root(paths, cfg) / args.split
    targets = [feature_root / f"{safe_feature_name(str(row['video']))}.pt" for row in rows]
    for target in [graph_path, *targets]:
        if target.exists():
            raise FileExistsError(f"refusing to overwrite event export: {target}")
    predictor = EventPredictor(
        args.checkpoint,
        dinov3_repo=_path(args, paths, "dinov3_repo"),
        dinov3_weights=_path(args, paths, "dinov3_weights"),
        device=args.device,
    )
    graphs = []
    for row, target in zip(rows, targets, strict=True):
        rally_id = str(row["video"])
        frame_dir = paths["frame_root"] / rally_id
        frames = list_images(frame_dir)
        if len(frames) != int(row["num_frames"]) or not frames:
            raise ValueError(f"frame count mismatch for {rally_id}")
        track_path = paths["ball_track_root"] / args.split / f"{rally_id}.json"
        if not track_path.is_file():
            track_path = track_path.with_suffix(".csv")
        fps = float(row["fps"])
        video_size = (int(row["width"]), int(row["height"])) if row.get("width") and row.get("height") else None
        output = predictor.predict(frames, fps=fps, ball_track=load_track_payload(track_path), video_size=video_size)
        graph = build_predicted_graph(
            output.events, rally_id=rally_id, frames_dir=frame_dir, fps=fps,
            num_frames=len(frames), width=row.get("width"), height=row.get("height"), split=args.split,
        )
        if not graph["strokes"]:
            raise ValueError(f"event export has no hit nodes: {rally_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shot_feature_payload(graph, output.frame_indices, output.frame_features, split=args.split), target)
        graphs.append(graph)
    write_jsonl(graph_path, graphs)
    return _emit({"graph_path": str(graph_path), "feature_root": str(feature_root), "rallies": len(graphs)}, args.output)


def predict(args: argparse.Namespace) -> int:
    from rallymotionreasoner.pipeline import RallyMotionReasoner

    paths = load_paths(args.config)
    model = RallyMotionReasoner(
        event_checkpoint=args.event_checkpoint,
        rgr_checkpoint=args.rgr_checkpoint,
        qwen_model=args.qwen_model or paths["qwen3_vl_model"],
        qwen_adapter=args.qwen_adapter,
        dinov3_repo=_path(args, paths, "dinov3_repo"),
        dinov3_weights=_path(args, paths, "dinov3_weights"),
        device=args.device,
    )
    payload = model.predict(args.video, args.question, fps=args.fps, ball_track=args.ball_track)
    return _emit(payload, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rallymotionreasoner", description="Stroke-evidence-grounded tennis video reasoning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="validate TRACE splits and build graph/QA artifacts")
    prepare.add_argument("--source-root", type=Path)
    prepare.add_argument("--output-root", type=Path)
    prepare.add_argument("--frame-root", type=Path)
    prepare.add_argument("--allow-missing-media", action="store_true")
    prepare.add_argument("--output", type=Path)
    prepare.set_defaults(handler=prepare_data)

    events = commands.add_parser("predict-events", help="run trajectory-region event detection")
    events.add_argument("--video", type=Path, required=True)
    events.add_argument("--checkpoint", type=Path, required=True, help="directory with trajectory_expert.pt and visual_expert.pt")
    events.add_argument("--dinov3-repo", type=Path)
    events.add_argument("--dinov3-weights", type=Path)
    events.add_argument("--ball-track", type=Path, required=True, help="TrackNet JSON payload or external CSV")
    events.add_argument("--device")
    events.add_argument("--fps", type=float)
    events.add_argument("--output", type=Path)
    events.set_defaults(handler=predict_events)

    export = commands.add_parser("export-events", help="export region event graphs and hit features for RGR")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--split", choices=["train", "val", "test"], required=True)
    export.add_argument("--experiment-config", type=Path, default=Path("configs/rallymotionreasoner.yaml"))
    export.add_argument("--dinov3-repo", type=Path)
    export.add_argument("--dinov3-weights", type=Path)
    export.add_argument("--device")
    export.add_argument("--output", type=Path)
    export.set_defaults(handler=export_events)

    full = commands.add_parser("predict", help="run event detection -> RGR -> grounded Qwen3-VL generation")
    full.add_argument("--video", type=Path, required=True)
    full.add_argument("--question", required=True)
    full.add_argument("--event-checkpoint", type=Path, required=True, help="directory with trajectory_expert.pt and visual_expert.pt")
    full.add_argument("--rgr-checkpoint", type=Path, required=True)
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
