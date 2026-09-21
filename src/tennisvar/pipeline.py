from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tennisvar.event_parsing.graph import build_predicted_graph
from tennisvar.generation.manifest import MANIFEST_NAME, file_sha256
from tennisvar.media import materialize_video
from tennisvar.video import local_indices, uniform_indices

OUTPUT_SCHEMA = "tennisvar.answer.v1"


def _config_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class TennisVAR:
    def __init__(
        self,
        *,
        event_checkpoint: Path,
        tgtr_checkpoint: Path,
        qwen_model: Path,
        dinov3_repo: Path,
        dinov3_weights: Path,
        qwen_adapter: Path | None = None,
        device: str | None = None,
    ) -> None:
        from tennisvar.event_parsing.runtime import build_event_predictor
        from tennisvar.generation.qwen import QwenVideoBackend
        from tennisvar.tactical_reasoning.runtime import TGTRCheckpointSelector

        self.event = build_event_predictor(
            event_checkpoint,
            dinov3_repo=dinov3_repo,
            dinov3_weights=dinov3_weights,
            device=device,
        )
        self.selector = TGTRCheckpointSelector(tgtr_checkpoint, device=device, event_backend=self.event.backend)
        if self.event.backend == "f3ed":
            if self.selector.state.get("upstream_event_checkpoint_sha256") != file_sha256(Path(event_checkpoint)):
                raise ValueError("TGTR checkpoint was trained from a different F3ED checkpoint")
            if self.selector.state.get("upstream_event_data_fingerprint") != self.event.checkpoint.get("data_fingerprint"):
                raise ValueError("TGTR/F3ED data fingerprints do not match")
        self.qwen = QwenVideoBackend(qwen_model, adapter=qwen_adapter)
        self.paths = {
            "event_checkpoint": str(Path(event_checkpoint)),
            "tgtr_checkpoint": str(Path(tgtr_checkpoint)),
            "qwen_model": str(Path(qwen_model)),
            "qwen_adapter": str(Path(qwen_adapter)) if qwen_adapter else None,
        }
        qwen_config = Path(qwen_model) / "config.json"
        self.artifact_hashes = {
            "event_checkpoint_sha256": file_sha256(Path(event_checkpoint)) if Path(event_checkpoint).is_file() else None,
            "tgtr_checkpoint_sha256": file_sha256(Path(tgtr_checkpoint)),
            "qwen_model_config_sha256": file_sha256(qwen_config),
            "qwen_adapter_manifest_sha256": (
                file_sha256(Path(qwen_adapter) / MANIFEST_NAME) if qwen_adapter else None
            ),
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
            if isinstance(ball_track, (str, Path)):
                ball_track = json.loads(Path(ball_track).read_text(encoding="utf-8"))
            runtime = self.event.predict(media.frame_paths, fps=media.fps, ball_track=ball_track)
            events = runtime.events
            graph_source = "f3ed_predicted"
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
                raise RuntimeError("TGTR produced no evidence candidates")
            frame_selection = set(uniform_indices(len(media.frame_paths), max_global_frames))
            for candidate in candidates:
                frame_selection.update(
                    local_indices(int(candidate["frame"]), len(media.frame_paths), radius=8, k=local_frames_per_candidate)
                )
            selected_frames = [media.frame_paths[index] for index in sorted(frame_selection)]
            answer = self.qwen.generate(selected_frames, question, candidates)
            by_id = {int(item["shot_id"]): item for item in candidates}
            evidence = []
            selected_candidates = [item for item in candidates if item.get("selected")]
            if not selected_candidates:
                selected_candidates = candidates[:1]
            for candidate in selected_candidates:
                shot_id = candidate["shot_id"]
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
            key_action_shot_ids = [
                int(item["shot_id"])
                for item in evidence
                if item.get("key_action_confidence", 0.0) >= 0.5
            ]
            return {
                "schema_version": OUTPUT_SCHEMA,
                "question": question,
                "answer": str(answer.get("answer") or ""),
                "answer_type": answer.get("answer_type", "free_form"),
                "level_1": self.selector.last_predictions.get("level_1"),
                "level_2": self.selector.last_predictions.get("level_2"),
                "level_3": self.selector.last_predictions.get("level_3"),
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
                    "graph_source": graph_source,
                    "event_feature_backend": runtime.feature_provenance.backend,
                    "event_detector_backend": self.event.backend,
                    "event_feature_weights_sha256": runtime.feature_provenance.weights_sha256,
                    "tracknet_checkpoint_sha256": runtime.feature_provenance.tracknet_checkpoint_sha256,
                    "tracknet_source": runtime.feature_provenance.tracknet_source,
                    "event_checkpoint_data_fingerprint": runtime.checkpoint_provenance.get("data_fingerprint"),
                    "event_checkpoint_sha256": runtime.checkpoint_provenance.get("checkpoint_sha256"),
                    "evidence_selector": self.selector.name,
                    "qwen_adapter_track": (
                        self.qwen.adapter_manifest.get("track") if self.qwen.adapter_manifest else None
                    ),
                    **self.paths,
                    **self.artifact_hashes,
                    "config_hash": _config_hash(self.paths),
                },
            }
