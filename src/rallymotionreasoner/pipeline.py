from __future__ import annotations

from pathlib import Path
from typing import Any

from rallymotionreasoner.event_detection.graph import build_predicted_graph
from rallymotionreasoner.features.ball_trajectory import load_track_payload
from rallymotionreasoner.media import materialize_video
from rallymotionreasoner.video import local_indices, uniform_indices

OUTPUT_SCHEMA = "rallymotionreasoner.answer.v1"


def select_frame_indices(
    num_frames: int,
    candidates: list[dict[str, Any]],
    *,
    max_global_frames: int,
    local_frames_per_candidate: int,
    max_frames: int = 32,
) -> list[int]:
    """Keep every evidence center, then global context and local detail within the video budget."""
    if num_frames <= 0 or max_frames <= 0:
        return []
    centers = [max(0, min(num_frames - 1, int(item["frame"]))) for item in candidates]
    selected = set(centers[:max_frames])
    selected.update(uniform_indices(num_frames, min(max_global_frames, max_frames - len(selected))))
    for center in centers:
        for index in local_indices(center, num_frames, radius=8, k=local_frames_per_candidate):
            if len(selected) >= max_frames:
                return sorted(selected)
            selected.add(index)
    return sorted(selected)


class RallyMotionReasoner:
    def __init__(
        self,
        *,
        event_checkpoint: Path,
        rgr_checkpoint: Path,
        qwen_model: Path,
        dinov3_repo: Path,
        dinov3_weights: Path,
        qwen_adapter: Path | None = None,
        device: str | None = None,
    ) -> None:
        from rallymotionreasoner.event_detection.runtime import EventPredictor
        from rallymotionreasoner.generation.qwen import QwenVideoBackend
        from rallymotionreasoner.graph_reasoning.runtime import RGRCheckpointSelector

        if not Path(event_checkpoint).is_dir():
            raise ValueError("event checkpoint must be a region expert directory")
        self.event = EventPredictor(
            event_checkpoint,
            dinov3_repo=dinov3_repo,
            dinov3_weights=dinov3_weights,
            device=device,
        )
        self.selector = RGRCheckpointSelector(rgr_checkpoint, device=device, event_backend=self.event.backend)
        self.qwen = QwenVideoBackend(qwen_model, adapter=qwen_adapter)
        self.paths = {
            "event_checkpoint": str(Path(event_checkpoint)),
            "rgr_checkpoint": str(Path(rgr_checkpoint)),
            "qwen_model": str(Path(qwen_model)),
            "qwen_adapter": str(Path(qwen_adapter)) if qwen_adapter else None,
        }

    def predict(
        self,
        video: str | Path,
        question: str,
        *,
        fps: float | None = None,
        max_global_frames: int = 16,
        local_frames_per_candidate: int = 3,
        ball_track: str | Path | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not str(question).strip():
            raise ValueError("question must be non-empty")
        with materialize_video(video, fps=fps) as media:
            if isinstance(ball_track, str | Path):
                ball_track = load_track_payload(Path(ball_track))
            runtime = self.event.predict(media.frame_paths, fps=media.fps, ball_track=ball_track)
            events = runtime.events
            graph_source = "motion_region_predicted"
            rally_id = Path(video).stem if Path(video).is_file() else Path(video).name
            graph = build_predicted_graph(
                events,
                rally_id=rally_id,
                frames_dir=Path(media.frame_paths[0]).parent,
                fps=media.fps,
                num_frames=len(media.frame_paths),
            )
            if not graph.get("strokes"):
                raise RuntimeError("event detector produced no hit candidates; strict predicted mode does not synthesize fallback events")
            graph["graph_source"] = graph_source
            candidates = self.selector.select(graph, question, runtime.frame_features, runtime.frame_indices)
            if not candidates:
                raise RuntimeError("RGR produced no evidence candidates")
            candidates = [candidate for candidate in candidates if candidate["selected"]]
            if candidates:
                frame_selection = select_frame_indices(
                    len(media.frame_paths),
                    candidates,
                    max_global_frames=max_global_frames,
                    local_frames_per_candidate=local_frames_per_candidate,
                )
                selected_frames = [media.frame_paths[index] for index in frame_selection]
                answer = self.qwen.generate(
                    selected_frames, question, candidates, frame_indices=frame_selection, fps=media.fps
                )
            else:
                answer = {
                    "answer": "",
                    "answerability": "unanswerable",
                    "explanation": "No event met the RGR evidence threshold.",
                    "causal_strength": "insufficient",
                }
            by_id = {int(item["shot_id"]): item for item in candidates}
            evidence = []
            for shot_id in answer.get("evidence_shot_ids", []):
                candidate = by_id.get(int(shot_id))
                if not candidate:
                    continue
                frame = int(candidate["frame"])
                evidence.append(
                    {
                        **candidate,
                        "start_sec": round(max(0, frame - 4) / media.fps, 4),
                        "end_sec": round(min(len(media.frame_paths) - 1, frame + 4) / media.fps, 4),
                    }
                )
            degraded = bool(answer.get("parse_error"))
            evidence_shot_ids = [int(item["shot_id"]) for item in evidence]
            key_action_shot_ids = [int(sid) for sid in answer.get("key_action_shot_ids", []) if int(sid) in evidence_shot_ids]
            return {
                "schema_version": OUTPUT_SCHEMA,
                "question": question,
                "answer": str(answer.get("answer") or ""),
                "answer_type": answer.get("answer_type", "free_form"),
                "level_1": answer.get("level_1"),
                "level_2": answer.get("level_2"),
                "level_3": answer.get("level_3"),
                "evidence_shot_ids": evidence_shot_ids,
                "key_action_shot_ids": key_action_shot_ids,
                "evidence_frames": [int(item["frame"]) for item in evidence],
                "answerability": answer.get("answerability", "unanswerable"),
                "explanation": str(answer.get("explanation") or ""),
                "observed_effect": answer.get("observed_effect", "unknown"),
                "causal_strength": answer.get("causal_strength", "insufficient"),
                "evidence": evidence,
                "provenance": {
                    "mode": "predicted",
                    "degraded": degraded,
                    "degraded_reason": answer.get("parse_error"),
                    "abstention_reason": "low_evidence" if not candidates else None,
                    "graph_source": graph_source,
                    "event_feature_backend": runtime.feature_provenance.backend,
                    "event_detector_backend": self.event.backend,
                    "tracknet_source": runtime.feature_provenance.tracknet_source,
                    "evidence_selector": self.selector.name,
                    "rgr_predictions": self.selector.last_predictions,
                    "qwen_adapter_track": (
                        self.qwen.adapter_manifest.get("track") if self.qwen.adapter_manifest else None
                    ),
                    **self.paths,
                },
            }
