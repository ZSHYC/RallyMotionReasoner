from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tennisvar.evaluation.matching import optimal_temporal_matching
from tennisvar.io import read_jsonl
from tennisvar.schema import ANSWER_TYPES, ANSWERABILITY, CAUSAL_STRENGTHS, OBSERVED_EFFECTS, answer_payload
from tennisvar.tactical_reasoning.graph_transformer import EDGE_TYPE_TO_ID

# Keep Latin words intact while giving CJK text character-level coverage.
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]")
NULL_LABEL = "null"


def tokenize(text: Any) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(str(text or ""))]


def read_graphs(path: Path) -> dict[str, dict[str, Any]]:
    return {row["rally_id"]: row for row in read_jsonl(path)}


def dataset_name(cfg: dict[str, Any] | None = None) -> str:
    data = (cfg or {}).get("data", {}) if isinstance(cfg, dict) else {}
    return str(data.get("dataset") or data.get("dataset_name") or "trace")


def dataset_root(paths: dict[str, Path], cfg: dict[str, Any] | None = None) -> Path:
    return paths["artifacts_root"] / dataset_name(cfg)


def dataset_relative_root(
    paths: dict[str, Path], cfg: dict[str, Any] | None, key: str, default: str
) -> Path:
    data = (cfg or {}).get("data", {}) if isinstance(cfg, dict) else {}
    relative = Path(str(data.get(key) or default))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"data.{key} must stay within the dataset artifact root")
    return dataset_root(paths, cfg) / relative


def graph_file(paths: dict[str, Path], split: str, source: str, cfg: dict[str, Any] | None = None) -> Path:
    name = dataset_name(cfg)
    if source == "gold":
        root = dataset_relative_root(paths, cfg, "graph_root_name", "graphs")
        return root / f"sgtr_graph_{name}_{split}.jsonl"
    if source == "region_fusion":
        root = dataset_relative_root(paths, cfg, "graph_root_name", "graphs_region_fusion")
        return root / f"sgtr_graph_region_fusion_{name}_{split}.jsonl"
    raise ValueError(f"unknown graph source: {source}")


def refs_file(paths: dict[str, Path], split: str, cfg: dict[str, Any] | None = None) -> Path:
    name = dataset_name(cfg)
    return dataset_root(paths, cfg) / "e2e_refs" / f"{name}_e2e_{split}.jsonl"


def visual_supervision_file(paths: dict[str, Path], cfg: dict[str, Any], split: str) -> Path | None:
    template = str(cfg.get("feature_extraction", {}).get("visual_supervision_path") or "")
    if not template:
        return None
    path = Path(template.format(split=split, dataset=dataset_name(cfg), artifacts_root=str(paths["artifacts_root"])))
    if not path.is_absolute():
        path = paths["project_root"] / path
    return path


def read_visual_supervision(path: Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        shots: dict[int, dict[str, Any]] = {}
        for stroke in row.get("strokes") or []:
            try:
                shots[int(stroke.get("shot_id"))] = stroke
            except Exception:
                continue
        rows[str(row.get("rally_id"))] = shots
    return rows


def feature_tokens_for_shot(shot: dict[str, Any], index: int, total: int, *, include_label_tokens: bool = True) -> list[str]:
    parsed = shot.get("parsed") or {}
    tokens = [f"shot_pos:{index + 1}", f"shot_count:{total}"]
    for prefix in ("before", "after"):
        gap = shot.get(f"bounce_{prefix}_gap")
        if gap is not None:
            tokens.append(f"bounce_{prefix}:1")
            tokens.append(f"bounce_{prefix}_gap:{min(int(gap) // 4, 8)}")
    if not include_label_tokens:
        return tokens
    for key in ["hitter", "court_zone", "phase", "hand", "technique", "direction", "outcome", "outcome_label"]:
        value = parsed.get(key) or shot.get(key)
        if value is not None:
            tokens.append(f"{key}:{str(value).lower()}")
    if parsed.get("approach") is not None:
        tokens.append(f"approach:{bool(parsed.get('approach'))}")
    label = shot.get("label")
    if label:
        tokens.extend(f"labeltok:{tok}" for tok in tokenize(label))
    return tokens


def question_tokens(row: dict[str, Any]) -> list[str]:
    # Metadata such as qa_type is absent at inference and made the training
    # and deployment question distributions differ. The question itself is
    # the only conditioning signal the reasoner can safely rely on.
    return [f"q:{tok}" for tok in tokenize(row.get("question"))]


def build_vocab(
    rows: list[dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    min_count: int = 1,
    *,
    include_label_tokens: bool = True,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tok in question_tokens(row):
            counts[tok] = counts.get(tok, 0) + 1
        graph = graphs.get(row.get("rally_id"), {})
        strokes = graph.get("strokes") or []
        for idx, shot in enumerate(strokes):
            for tok in feature_tokens_for_shot(shot, idx, len(strokes), include_label_tokens=include_label_tokens):
                counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, count in sorted(counts.items()):
        if count >= min_count:
            vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, 1) for tok in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))


def build_label_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    values: dict[str, set[str]] = {
        "answer_type": set(ANSWER_TYPES),
        "answerability": set(ANSWERABILITY),
        "causal_strength": set(CAUSAL_STRENGTHS),
        "observed_effect": set(OBSERVED_EFFECTS),
        "level_1": {NULL_LABEL, "A", "B", "C", "D", "E", "O"},
        "level_2": {NULL_LABEL},
        "level_3": {NULL_LABEL},
    }
    for row in rows:
        gold = answer_payload(row)
        for key in values:
            value = gold.get(key)
            values[key].add(str(value) if value is not None else NULL_LABEL)
    return {key: {value: idx for idx, value in enumerate(sorted(vals))} for key, vals in values.items()}


def valid_int_set(values: Any, valid: set[int]) -> set[int]:
    out: set[int] = set()
    if not isinstance(values, list):
        return out
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in valid:
            out.add(item)
    return out


def _frame_list(values: Any) -> list[int]:
    output: list[int] = []
    for value in values if isinstance(values, list) else []:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def match_gold_frames_to_predicted_shots(
    strokes: list[dict[str, Any]],
    gold_frames: Any,
    *,
    tolerance: int,
) -> set[int]:
    """One-to-one temporal alignment; never assumes gold/predicted shot ordinals coincide."""
    references = _frame_list(gold_frames)
    predicted = [int(stroke.get("frame") or 0) for stroke in strokes]
    return {
        int(strokes[pred_index].get("shot_id", pred_index + 1))
        for pred_index, _ in optimal_temporal_matching(predicted, references, tolerance)
    }


def inverse_maps(label_maps: dict[str, dict[str, int]]) -> dict[str, dict[int, str]]:
    return {key: {idx: value for value, idx in mapping.items()} for key, mapping in label_maps.items()}


def safe_feature_name(rally_id: str) -> str:
    return rally_id.replace("/", "__")


def load_visual_features(
    feature_root: Path | None,
    split: str,
    rally_id: str,
    shot_ids: list[int],
    feature_dim: int,
    *,
    strict: bool = False,
) -> list[list[float]]:
    zeros = [[0.0] * feature_dim for _ in shot_ids]
    if not feature_root:
        if strict:
            raise FileNotFoundError(f"visual feature root is required for rally {rally_id}")
        return zeros
    path = feature_root / split / f"{safe_feature_name(rally_id)}.pt"
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"missing visual features: {path}")
        return zeros
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"cannot load visual features {path}: {type(exc).__name__}: {exc}") from exc
        return zeros
    if strict:
        if (
            data.get("schema") != "tennisvar.tgtr_event_features"
            or data.get("source") != "region_fusion_predicted"
            or data.get("rally_id") != rally_id
            or data.get("split") != split
        ):
            raise ValueError(f"visual feature provenance contract mismatch: {path}")
    tensor = data.get("shot_features")
    cached_ids = data.get("shot_ids") or []
    if tensor is None:
        if strict:
            raise ValueError(f"visual feature file has no shot_features: {path}")
        return zeros
    if not isinstance(tensor, Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.ndim != 2 or int(tensor.shape[-1]) != feature_dim:
        message = f"visual feature dimension mismatch in {path}: {tuple(tensor.shape)} expected (*, {feature_dim})"
        if strict:
            raise ValueError(message)
        return zeros
    by_id = {int(sid): tensor[idx].float().tolist() for idx, sid in enumerate(cached_ids) if idx < tensor.shape[0]}
    missing_ids = [int(sid) for sid in shot_ids if int(sid) not in by_id]
    if strict and missing_ids:
        raise ValueError(f"visual feature cache {path} is missing shot ids: {missing_ids}")
    return [by_id.get(int(sid), [0.0] * feature_dim)[:feature_dim] for sid in shot_ids]


def load_motion_features(
    feature_root: Path | None,
    split: str,
    rally_id: str,
    shot_ids: list[int],
    motion_dim: int,
) -> list[list[float]]:
    if motion_dim <= 0:
        return [[0.0] * motion_dim for _ in shot_ids]
    if not feature_root:
        raise ValueError("motion feature root is required when motion features are enabled")
    path = feature_root / split / f"{safe_feature_name(rally_id)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing motion features: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    tensor = data.get("motion_stats")
    cached_ids = data.get("shot_ids") or []
    if tensor is None:
        raise ValueError(f"motion feature file has no motion_stats: {path}")
    if not isinstance(tensor, Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.ndim != 2 or int(tensor.shape[-1]) != motion_dim:
        raise ValueError(f"motion feature dimension mismatch in {path}: {tuple(tensor.shape)}")
    by_id = {int(sid): tensor[idx].float().tolist() for idx, sid in enumerate(cached_ids) if idx < tensor.shape[0]}
    missing_ids = [int(sid) for sid in shot_ids if int(sid) not in by_id]
    if missing_ids:
        raise ValueError(f"motion feature cache {path} is missing shot ids: {missing_ids}")
    return [by_id[int(sid)][:motion_dim] for sid in shot_ids]


@dataclass
class EncodedItem:
    qa_id: str
    rally_id: str
    question_ids: list[int]
    node_ids: list[list[int]]
    node_frames: list[float]
    visual_features: list[list[float]]
    motion_features: list[list[float]]
    ball_xy: list[list[float]]
    ball_mask: list[float]
    ball_visible: list[float]
    contact_frame: list[float]
    contact_mask: list[float]
    frame_scale: float
    edge_index: list[list[int]]
    edge_type: list[int]
    evidence_targets: list[float]
    key_targets: list[float]
    labels: dict[str, int]
    shot_ids: list[int]
    frame_by_shot: dict[int, int]
    question: str


class GraphQADataset(Dataset[EncodedItem]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        graphs: dict[str, dict[str, Any]],
        vocab: dict[str, int],
        label_maps: dict[str, dict[str, int]],
        *,
        split: str,
        feature_root: Path | None = None,
        visual_supervision: dict[str, dict[int, dict[str, Any]]] | None = None,
        visual_feature_dim: int = 48,
        motion_feature_dim: int = 0,
        include_label_tokens: bool = True,
        use_edges: bool = True,
        excluded_edge_types: list[str] | tuple[str, ...] = (),
        max_question_tokens: int = 64,
        max_node_tokens: int = 24,
        max_strokes: int = 32,
        require_visual_features: bool = False,
        graph_source: str = "region_fusion",
        evidence_frame_tolerance: int = 16,
    ) -> None:
        self.items: list[EncodedItem] = []
        self.missing_graphs = 0
        for row in rows:
            rally_id = str(row.get("rally_id"))
            graph = graphs.get(rally_id, {})
            strokes = (graph.get("strokes") or [])[:max_strokes]
            if not strokes:
                self.missing_graphs += 1
                continue
            gold = answer_payload(row)
            shot_ids = [int(shot.get("shot_id", idx + 1)) for idx, shot in enumerate(strokes)]
            valid = set(shot_ids)
            frame_by_shot = {int(shot.get("shot_id", idx + 1)): int(shot.get("frame") or 0) for idx, shot in enumerate(strokes)}
            node_ids = [
                encode_tokens(
                    feature_tokens_for_shot(shot, idx, len(strokes), include_label_tokens=include_label_tokens),
                    vocab,
                    max_node_tokens,
                )
                for idx, shot in enumerate(strokes)
            ]
            max_frame = max([int(shot.get("frame") or 0) for shot in strokes] + [1])
            node_frames = [float(int(shot.get("frame") or 0)) / max_frame for shot in strokes]
            sid_to_node = {sid: idx for idx, sid in enumerate(shot_ids)}
            sup_by_shot = (visual_supervision or {}).get(rally_id, {})
            ball_xy: list[list[float]] = []
            ball_mask: list[float] = []
            ball_visible: list[float] = []
            contact_frame: list[float] = []
            contact_mask: list[float] = []
            for sid in shot_ids:
                sup = sup_by_shot.get(int(sid), {})
                ball = sup.get("ball") or {}
                image_size = float(ball.get("image_size") or 224.0)
                x = ball.get("x")
                y = ball.get("y")
                if x is not None and y is not None and image_size > 1:
                    ball_xy.append([float(x) / image_size, float(y) / image_size])
                    ball_mask.append(1.0)
                else:
                    ball_xy.append([0.0, 0.0])
                    ball_mask.append(0.0)
                confidence = ball.get("confidence")
                if confidence is None:
                    confidence = 1.0 if ball.get("visible") else 0.0
                ball_visible.append(max(0.0, min(1.0, float(confidence))))
                contact = sup.get("contact") or {}
                contact_value = contact.get("frame")
                if contact_value is not None and max_frame > 0:
                    contact_frame.append(float(contact_value) / max_frame)
                    contact_mask.append(1.0)
                else:
                    contact_frame.append(0.0)
                    contact_mask.append(0.0)
            edge_index: list[list[int]] = []
            edge_type: list[int] = []
            excluded_edges = {str(value) for value in excluded_edge_types}
            if use_edges:
                for edge in graph.get("edges") or []:
                    relation = str(edge.get("type"))
                    if relation in excluded_edges or relation not in EDGE_TYPE_TO_ID:
                        continue
                    src = str(edge.get("source", ""))
                    dst = str(edge.get("target", ""))
                    if not src.startswith("shot_") or not dst.startswith("shot_"):
                        continue
                    src_id = int(src.split("_")[-1])
                    dst_id = int(dst.split("_")[-1])
                    if src_id in sid_to_node and dst_id in sid_to_node:
                        edge_index.append([sid_to_node[src_id], sid_to_node[dst_id]])
                        edge_type.append(EDGE_TYPE_TO_ID[relation])
            if graph_source == "region_fusion":
                evidence_ids = match_gold_frames_to_predicted_shots(
                    strokes, gold.get("evidence_frames"), tolerance=evidence_frame_tolerance
                )
                key_ids = match_gold_frames_to_predicted_shots(
                    strokes,
                    row.get("key_action_frames", gold.get("key_action_frames")),
                    tolerance=evidence_frame_tolerance,
                )
            else:
                evidence_ids = valid_int_set(gold.get("evidence_shot_ids"), valid)
                key_ids = valid_int_set(gold.get("key_action_shot_ids"), valid)
            labels = {}
            for key, mapping in label_maps.items():
                value = gold.get(key)
                labels[key] = mapping.get(str(value) if value is not None else NULL_LABEL, mapping.get(NULL_LABEL, 0))
            self.items.append(
                EncodedItem(
                    qa_id=str(row["qa_id"]),
                    rally_id=rally_id,
                    question_ids=encode_tokens(question_tokens(row), vocab, max_question_tokens),
                    node_ids=node_ids,
                    node_frames=node_frames,
                    visual_features=load_visual_features(
                        feature_root,
                        split,
                        rally_id,
                        shot_ids,
                        visual_feature_dim,
                        strict=require_visual_features,
                    ),
                    motion_features=load_motion_features(feature_root, split, rally_id, shot_ids, motion_feature_dim),
                    ball_xy=ball_xy,
                    ball_mask=ball_mask,
                    ball_visible=ball_visible,
                    contact_frame=contact_frame,
                    contact_mask=contact_mask,
                    frame_scale=float(max_frame),
                    edge_index=edge_index,
                    edge_type=edge_type,
                    evidence_targets=[1.0 if sid in evidence_ids else 0.0 for sid in shot_ids],
                    key_targets=[1.0 if sid in key_ids else 0.0 for sid in shot_ids],
                    labels=labels,
                    shot_ids=shot_ids,
                    frame_by_shot=frame_by_shot,
                    question=str(row.get("question") or ""),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedItem:
        return self.items[idx]


def collate(items: list[EncodedItem]) -> dict[str, Any]:
    batch = len(items)
    max_nodes = max(len(item.node_ids) for item in items)
    node_token_len = len(items[0].node_ids[0])
    question_len = len(items[0].question_ids)
    visual_dim = len(items[0].visual_features[0]) if items[0].visual_features else 0
    motion_dim = len(items[0].motion_features[0]) if items[0].motion_features else 0
    max_edges = max([len(item.edge_index) for item in items] + [1])
    question_ids = torch.zeros(batch, question_len, dtype=torch.long)
    node_ids = torch.zeros(batch, max_nodes, node_token_len, dtype=torch.long)
    node_frames = torch.zeros(batch, max_nodes, dtype=torch.float32)
    visual_features = torch.zeros(batch, max_nodes, visual_dim, dtype=torch.float32)
    motion_features = torch.zeros(batch, max_nodes, motion_dim, dtype=torch.float32)
    ball_xy = torch.zeros(batch, max_nodes, 2, dtype=torch.float32)
    ball_mask = torch.zeros(batch, max_nodes, dtype=torch.float32)
    ball_visible = torch.zeros(batch, max_nodes, dtype=torch.float32)
    contact_frame = torch.zeros(batch, max_nodes, dtype=torch.float32)
    contact_mask = torch.zeros(batch, max_nodes, dtype=torch.float32)
    frame_scale = torch.ones(batch, dtype=torch.float32)
    node_mask = torch.zeros(batch, max_nodes, dtype=torch.bool)
    edge_index = torch.zeros(batch, max_edges, 2, dtype=torch.long)
    edge_type = torch.zeros(batch, max_edges, dtype=torch.long)
    edge_mask = torch.zeros(batch, max_edges, dtype=torch.bool)
    evidence = torch.zeros(batch, max_nodes, dtype=torch.float32)
    key = torch.zeros(batch, max_nodes, dtype=torch.float32)
    labels = {name: torch.zeros(batch, dtype=torch.long) for name in items[0].labels}
    for b, item in enumerate(items):
        n = len(item.node_ids)
        question_ids[b] = torch.tensor(item.question_ids, dtype=torch.long)
        node_ids[b, :n] = torch.tensor(item.node_ids, dtype=torch.long)
        node_frames[b, :n] = torch.tensor(item.node_frames, dtype=torch.float32)
        if visual_dim:
            visual_features[b, :n] = torch.tensor(item.visual_features, dtype=torch.float32)
        if motion_dim:
            motion_features[b, :n] = torch.tensor(item.motion_features, dtype=torch.float32)
        ball_xy[b, :n] = torch.tensor(item.ball_xy, dtype=torch.float32)
        ball_mask[b, :n] = torch.tensor(item.ball_mask, dtype=torch.float32)
        ball_visible[b, :n] = torch.tensor(item.ball_visible, dtype=torch.float32)
        contact_frame[b, :n] = torch.tensor(item.contact_frame, dtype=torch.float32)
        contact_mask[b, :n] = torch.tensor(item.contact_mask, dtype=torch.float32)
        frame_scale[b] = float(item.frame_scale)
        node_mask[b, :n] = True
        evidence[b, :n] = torch.tensor(item.evidence_targets, dtype=torch.float32)
        key[b, :n] = torch.tensor(item.key_targets, dtype=torch.float32)
        if item.edge_index:
            e = len(item.edge_index)
            edge_index[b, :e] = torch.tensor(item.edge_index, dtype=torch.long)
            edge_type[b, :e] = torch.tensor(item.edge_type, dtype=torch.long)
            edge_mask[b, :e] = True
        for name, value in item.labels.items():
            labels[name][b] = value
    return {
        "question_ids": question_ids,
        "node_ids": node_ids,
        "node_frames": node_frames,
        "visual_features": visual_features,
        "motion_features": motion_features,
        "ball_xy": ball_xy,
        "ball_mask": ball_mask,
        "ball_visible": ball_visible,
        "contact_frame": contact_frame,
        "contact_mask": contact_mask,
        "frame_scale": frame_scale,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_mask": edge_mask,
        "evidence": evidence,
        "key": key,
        "labels": labels,
        "items": items,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if key == "items":
            out[key] = value
        elif key == "labels":
            out[key] = {name: tensor.to(device) for name, tensor in value.items()}
        else:
            out[key] = value.to(device)
    return out


def make_graph_datasets(
    paths: dict[str, Path],
    cfg: dict[str, Any],
    feature_root: Path | None,
) -> tuple[GraphQADataset, GraphQADataset, dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    graph_source = str(data_cfg.get("graph_source", "region_fusion"))
    include_label_tokens = bool(data_cfg.get("include_label_tokens", True))
    use_edges = bool(data_cfg.get("use_edges", True))
    excluded_edge_types = [str(value) for value in data_cfg.get("excluded_edge_types", [])]
    train_rows = read_jsonl(refs_file(paths, "train", cfg))
    val_rows = read_jsonl(refs_file(paths, "val", cfg))
    label_map_source = str(data_cfg.get("label_map_source", "train"))
    if label_map_source == "train":
        label_rows = train_rows
    elif label_map_source == "train_val":
        label_rows = train_rows + val_rows
    else:
        raise ValueError("data.label_map_source must be 'train' or 'train_val'; using test labels is not allowed")
    train_graphs = read_graphs(graph_file(paths, "train", graph_source, cfg))
    val_graphs = read_graphs(graph_file(paths, "val", graph_source, cfg))
    train_visual = read_visual_supervision(visual_supervision_file(paths, cfg, "train"))
    val_visual = read_visual_supervision(visual_supervision_file(paths, cfg, "val"))
    vocab = build_vocab(train_rows, train_graphs, min_count=int(data_cfg.get("min_count", 1)), include_label_tokens=include_label_tokens)
    label_maps = build_label_maps(label_rows)
    common = {
        "feature_root": feature_root,
        "visual_feature_dim": int(cfg.get("feature_extraction", {}).get("feature_dim", 800)),
        "motion_feature_dim": int(cfg.get("feature_extraction", {}).get("motion_feature_dim", 0)),
        "include_label_tokens": include_label_tokens,
        "use_edges": use_edges,
        "excluded_edge_types": excluded_edge_types,
        "max_question_tokens": int(data_cfg.get("max_question_tokens", 64)),
        "max_node_tokens": int(data_cfg.get("max_node_tokens", 24)),
        "max_strokes": int(data_cfg.get("max_strokes", 32)),
        "require_visual_features": bool(data_cfg.get("require_visual_features", True)),
        "graph_source": graph_source,
        "evidence_frame_tolerance": int(data_cfg.get("evidence_frame_tolerance", 16)),
    }
    train_ds = GraphQADataset(train_rows, train_graphs, vocab, label_maps, split="train", visual_supervision=train_visual, **common)
    val_ds = GraphQADataset(val_rows, val_graphs, vocab, label_maps, split="val", visual_supervision=val_visual, **common)
    meta = {
        "vocab": vocab,
        "label_maps": label_maps,
        "graph_source": graph_source,
        "dataset": dataset_name(cfg),
        "train_graph_count": len(train_graphs),
        "val_graph_count": len(val_graphs),
        "feature_root": str(feature_root) if feature_root else None,
        "include_label_tokens": include_label_tokens,
        "use_edges": use_edges,
        "excluded_edge_types": excluded_edge_types,
        "label_map_source": label_map_source,
        "test_label_leakage": False,
        "evidence_alignment": "one_to_one_temporal" if graph_source == "region_fusion" else "gold_shot_id",
    }
    return train_ds, val_ds, meta
