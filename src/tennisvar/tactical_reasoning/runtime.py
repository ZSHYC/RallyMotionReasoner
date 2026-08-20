from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from tennisvar.checkpoint_validation import validate_tgtr_checkpoint
from tennisvar.data import GraphQADataset, collate, move_batch
from tennisvar.event_parsing.dataset import safe_name
from tennisvar.tactical_reasoning.model import TacticalGraphGuidedTemporalReasoner


def runtime_feature_payload(
    *,
    rally_id: str,
    shot_ids: list[int],
    shot_frames: list[int],
    shot_features: Any,
    event_checkpoint_data_fingerprint: str,
    event_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the same strict TGTR feature contract used by split export."""
    return {
        "schema": "tennisvar.tgtr_event_features.v2",
        "source": "f3ed_predicted",
        "rally_id": rally_id,
        "split": "inference",
        "shot_ids": shot_ids,
        "shot_frames": shot_frames,
        "shot_features": shot_features,
        "event_checkpoint_data_fingerprint": event_checkpoint_data_fingerprint,
        "event_checkpoint_sha256": event_checkpoint_sha256,
    }


class TGTRCheckpointSelector:
    name = "tgtr_checkpoint_predicted_events"

    def __init__(self, checkpoint: Path, *, device: str | None = None, top_k: int = 8) -> None:
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint)
        self.state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        validate_tgtr_checkpoint(self.state, expected_graph_source="f3ed")
        cfg = self.state.get("config") or {}
        if str(self.state.get("graph_source") or cfg.get("data", {}).get("graph_source")) != "f3ed":
            raise ValueError("TGTR selector requires a checkpoint trained on predicted F3ED graphs")
        if bool(cfg.get("data", {}).get("include_label_tokens", True)):
            raise ValueError("TGTR main selector refuses checkpoints that consume event label text")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_k = int(top_k)
        self.model = TacticalGraphGuidedTemporalReasoner(
            len(self.state["vocab"]),
            self.state["label_maps"],
            visual_feature_dim=int(self.state["visual_feature_dim"]),
            hidden_dim=int(self.state.get("hidden_dim", 256)),
            num_layers=int(self.state.get("num_layers", 2)),
            num_heads=int(self.state.get("num_heads", 8)),
            dropout=float(self.state.get("dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(self.state["model_state"], strict=True)
        self.model.eval()

    def select(
        self,
        graph: dict[str, Any],
        question: str,
        frame_features: Any,
        frame_indices: list[int],
    ) -> list[dict[str, Any]]:
        strokes = graph.get("strokes") or []
        if not strokes:
            return []
        expected_dim = int(self.state["visual_feature_dim"])
        if int(frame_features.shape[-1]) != expected_dim:
            raise ValueError(f"TGTR/event feature dimension mismatch: {frame_features.shape[-1]} != {expected_dim}")
        selected = []
        for stroke in strokes:
            frame = int(stroke["frame"])
            index = min(range(len(frame_indices)), key=lambda item: abs(frame_indices[item] - frame))
            selected.append(frame_features[index])
        rally_id = str(graph["rally_id"])
        cfg = self.state.get("config") or {}
        with tempfile.TemporaryDirectory(prefix="tennisvar_tgtr_") as temp:
            root = Path(temp)
            target = root / "inference" / f"{safe_name(rally_id)}.pt"
            target.parent.mkdir(parents=True)
            self.torch.save(
                runtime_feature_payload(
                    rally_id=rally_id,
                    shot_ids=[int(stroke["shot_id"]) for stroke in strokes],
                    shot_frames=[int(stroke["frame"]) for stroke in strokes],
                    shot_features=self.torch.stack(selected),
                    event_checkpoint_data_fingerprint=str(
                        self.state["upstream_event_data_fingerprint"]
                    ),
                    event_checkpoint_sha256=str(self.state["upstream_event_checkpoint_sha256"]),
                ),
                target,
            )
            row = {
                "qa_id": f"inference:{rally_id}",
                "rally_id": rally_id,
                "qa_type": "inference",
                "qa_group": "inference",
                "question": question,
                "gold_answer": {},
            }
            dataset = GraphQADataset(
                [row],
                {rally_id: graph},
                self.state["vocab"],
                self.state["label_maps"],
                split="inference",
                feature_root=root,
                visual_feature_dim=expected_dim,
                motion_feature_dim=0,
                include_label_tokens=False,
                use_edges=bool(cfg.get("data", {}).get("use_edges", True)),
                require_visual_features=True,
            )
            if len(dataset) != 1:
                raise RuntimeError("TGTR could not encode the predicted event graph")
            batch = move_batch(collate([dataset[0]]), self.device)
            with self.torch.no_grad():
                output = self.model(batch)
            evidence = self.torch.sigmoid(output["evidence_logits"][0, : len(strokes)]).cpu().tolist()
            key_scores = self.torch.sigmoid(output["key_action_logits"][0, : len(strokes)]).cpu().tolist()
        order = sorted(range(len(strokes)), key=lambda index: evidence[index], reverse=True)[: self.top_k]
        candidates = []
        for index in order:
            stroke = strokes[index]
            parsed = stroke.get("parsed") or {}
            candidates.append(
                {
                    "shot_id": int(stroke["shot_id"]),
                    "frame": int(stroke["frame"]),
                    "time_sec": stroke.get("time_sec"),
                    "confidence": float(evidence[index]),
                    "key_action_confidence": float(key_scores[index]),
                    "event_confidence": float(stroke.get("confidence", 0.0)),
                    "hitter": parsed.get("hitter"),
                    "court_zone": parsed.get("court_zone"),
                    "phase": parsed.get("phase"),
                    "hand": parsed.get("hand"),
                    "technique": parsed.get("technique"),
                    "direction": parsed.get("direction"),
                    "approach": parsed.get("approach"),
                    "outcome": parsed.get("outcome"),
                    "attribute_confidence": stroke.get("attribute_confidence") or {},
                    "source": "tgtr_predicted",
                }
            )
        return candidates
