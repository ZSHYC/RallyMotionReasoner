from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from rallymotionreasoner.checkpoint_validation import validate_rgr_checkpoint
from rallymotionreasoner.data import GraphQADataset, collate, move_batch
from rallymotionreasoner.data.graph_qa import safe_feature_name
from rallymotionreasoner.event_detection.features import shot_feature_payload
from rallymotionreasoner.graph_reasoning.model import RallyGraphReasoner


class RGRCheckpointSelector:
    name = "rgr_checkpoint_predicted_events"

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str | None = None,
        top_k: int = 8,
        event_backend: str = "motion_region",
    ) -> None:
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint)
        self.state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        validate_rgr_checkpoint(
            self.state,
            expected_graph_source="motion_region",
            expected_event_backend=event_backend,
        )
        cfg = self.state.get("config") or {}
        self.event_backend = str(event_backend)
        if bool(cfg.get("data", {}).get("include_label_tokens", True)):
            raise ValueError("RGR main selector refuses checkpoints that consume event label text")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_k = int(top_k)
        self.evidence_threshold = float(self.state["thresholds"]["evidence_threshold"])
        self.last_predictions: dict[str, str | None] = {}
        model_cfg = self.state.get("config", {}).get("feature_extraction", {})
        self.motion_feature_dim = int(self.state.get("motion_feature_dim", model_cfg.get("motion_feature_dim", 0)))
        self.model = RallyGraphReasoner(
            len(self.state["vocab"]),
            self.state["label_maps"],
            visual_feature_dim=int(self.state["visual_feature_dim"]),
            motion_feature_dim=self.motion_feature_dim,
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
            raise ValueError(f"RGR/event feature dimension mismatch: {frame_features.shape[-1]} != {expected_dim}")
        rally_id = str(graph["rally_id"])
        cfg = self.state.get("config") or {}
        with tempfile.TemporaryDirectory(prefix="rallymotionreasoner_rgr_") as temp:
            root = Path(temp)
            target = root / "inference" / f"{safe_feature_name(rally_id)}.pt"
            target.parent.mkdir(parents=True)
            self.torch.save(
                shot_feature_payload(graph, frame_indices, frame_features, split="inference"),
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
                motion_feature_dim=self.motion_feature_dim,
                include_label_tokens=False,
                use_edges=bool(cfg.get("data", {}).get("use_edges", True)),
                require_visual_features=True,
                max_strokes=len(strokes),
                max_question_tokens=int(cfg.get("data", {}).get("max_question_tokens", 64)),
                max_node_tokens=int(cfg.get("data", {}).get("max_node_tokens", 24)),
            )
            if len(dataset) != 1:
                raise RuntimeError("RGR could not encode the predicted event graph")
            batch = move_batch(collate([dataset[0]]), self.device)
            with self.torch.no_grad():
                output = self.model(batch)
            evidence = self.torch.sigmoid(output["evidence_logits"][0, : len(strokes)]).cpu().tolist()
            key_scores = self.torch.sigmoid(output["key_action_logits"][0, : len(strokes)]).cpu().tolist()
            self.last_predictions = {}
            for name, mapping in self.state["label_maps"].items():
                key = f"{name}_logits"
                if key in output:
                    inverse = {int(index): str(label) for label, index in mapping.items()}
                    self.last_predictions[name] = inverse[int(output[key][0].argmax().item())]
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
                    "selected": bool(evidence[index] >= self.evidence_threshold),
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
                    "source": "rgr_predicted",
                }
            )
        return candidates
