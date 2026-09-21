from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .f3label import parse_label, stroke_doc

CLIP_RE = re.compile(r"^(.+)_([0-9]+)_([0-9]+)$")

LEVEL1_NAMES = {
    "A": "serve_patterns",
    "B": "return_patterns",
    "C": "baseline_construction",
    "D": "net_transition",
    "E": "opponent_net_response",
    "O": "no_primary_tactic",
}


def match_id_from_clip(clip: str) -> str:
    m = CLIP_RE.match(clip)
    return m.group(1) if m else clip


def clip_frame_range(clip: str) -> tuple[int | None, int | None]:
    m = CLIP_RE.match(clip)
    if not m:
        return None, None
    return int(m.group(2)), int(m.group(3))


def graph_paths(graphs_dir: Path) -> dict[str, Path]:
    return {split: graphs_dir / f"sgtr_graph_{split}.jsonl" for split in ["train", "val", "test"]}


def shot_ids_from_node(node: Any) -> list[int]:
    if not isinstance(node, dict):
        return []
    values = node.get("shot_ids")
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        if isinstance(value, int) or str(value).isdigit():
            out.append(int(value))
    return out


def followup_items(unit: dict[str, Any]) -> list[dict[str, Any]]:
    chain = unit.get("evidence_chain") or {}
    items = chain.get("observed_followup") or unit.get("observed_followup") or []
    return items if isinstance(items, list) else []


def setup_shot_ids(unit: dict[str, Any]) -> list[int]:
    return shot_ids_from_node((unit.get("evidence_chain") or {}).get("setup"))


def key_action_shot_ids(unit: dict[str, Any]) -> list[int]:
    return sorted(set(shot_ids_from_node((unit.get("evidence_chain") or {}).get("key_action"))))


def followup_shot_ids(unit: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for item in followup_items(unit):
        ids.extend(shot_ids_from_node(item))
    return sorted(set(ids))


def evidence_shot_ids(unit: dict[str, Any]) -> list[int]:
    ids = setup_shot_ids(unit) + key_action_shot_ids(unit) + followup_shot_ids(unit)
    return sorted(set(ids))


def observed_effect(unit: dict[str, Any]) -> str | None:
    items = followup_items(unit)
    if items:
        types = sorted({str(item.get("type")) for item in items if item.get("type")})
        if types:
            return "+".join(types)
    result = unit.get("tactic_result")
    if result in {"key_action_winner", "sequence_completion_winner"}:
        return "winner"
    if result == "induced_forced_error":
        return "opponent_forced_error"
    if result == "countered_by_opponent_winner":
        return "opponent_winner"
    if result in {"execution_forced_error", "execution_unforced_error"}:
        return str(result)
    if result in {"completed_no_terminal_effect", "no_verified_effect"}:
        return str(result)
    return str(result) if isinstance(result, str) else None


def causal_assessment(unit: dict[str, Any]) -> dict[str, Any]:
    result = unit.get("tactic_result")
    effect = observed_effect(unit)
    if result in {"key_action_winner", "sequence_completion_winner", "execution_forced_error", "execution_unforced_error"}:
        strength = "event_supported"
    elif result == "induced_forced_error":
        strength = "plausible_rule_based"
    elif result in {"completed_no_terminal_effect", "no_verified_effect"}:
        strength = "not_verified"
    elif result:
        strength = "insufficient"
    else:
        strength = "none"
    return {
        "observed_effect": effect,
        "causal_strength": strength,
        "causal_claim_allowed": strength in {"event_supported", "plausible_rule_based"},
    }


def build_strokes(record: dict[str, Any]) -> list[dict[str, Any]]:
    fps = float(record.get("fps") or 25.0)
    strokes: list[dict[str, Any]] = []
    for idx, event in enumerate(record.get("events") or [], start=1):
        frame = int(event.get("frame"))
        label = str(event.get("label"))
        parsed = parse_label(label, event.get("outcome"))
        shot = {
            "shot_id": idx,
            "frame": frame,
            "time_sec": round(frame / fps, 4) if fps else None,
            "label": label,
            "outcome": event.get("outcome"),
            "parsed": parsed,
        }
        shot["doc"] = stroke_doc(shot)
        strokes.append(shot)
    return strokes


def build_unit(unit: dict[str, Any], idx: int, strokes: list[dict[str, Any]]) -> dict[str, Any]:
    ev_ids = evidence_shot_ids(unit)
    key_ids = key_action_shot_ids(unit)
    frames_by_id = {s["shot_id"]: s["frame"] for s in strokes}
    labels_by_id = {s["shot_id"]: s["label"] for s in strokes}
    causal = causal_assessment(unit)
    level_1 = unit.get("level_1")
    out = {
        "unit_id": f"unit_{idx:03d}",
        "level_1": level_1,
        "level1_stage": LEVEL1_NAMES.get(str(level_1), level_1),
        "level_2": unit.get("level_2"),
        "level2_family": unit.get("level_2"),
        "level_3": unit.get("level_3"),
        "level3_tactic": unit.get("level_3"),
        "player": unit.get("player"),
        "is_primary_eval": bool(unit.get("is_primary_eval")),
        "tactic_result": unit.get("tactic_result"),
        "attributes": unit.get("attributes") or {},
        "evidence_chain": unit.get("evidence_chain") or {},
        "setup_shots": setup_shot_ids(unit),
        "key_action_shots": key_ids,
        "followup_shots": followup_shot_ids(unit),
        "evidence_shots": ev_ids,
        "observed_effect": causal["observed_effect"],
        "causal_strength": causal["causal_strength"],
        "causal_assessment": causal,
        "evidence_level": "E1_event_sequence",
        "gold_evidence": {
            "shot_ids": ev_ids,
            "key_action_shot_ids": key_ids,
            "frames": [frames_by_id[sid] for sid in ev_ids if sid in frames_by_id],
            "event_labels": [labels_by_id[sid] for sid in ev_ids if sid in labels_by_id],
        },
    }
    out["doc"] = unit_doc(out)
    return out


def unit_doc(unit: dict[str, Any]) -> str:
    return (
        f"unit_id={unit.get('unit_id')}; player={unit.get('player')}; "
        f"setup={unit.get('setup_shots')}; key={unit.get('key_action_shots')}; "
        f"followup={unit.get('followup_shots')}; result={unit.get('tactic_result')}; "
        f"effect={unit.get('observed_effect')}; causal_strength={unit.get('causal_strength')}"
    )


def build_edges(strokes: list[dict[str, Any]], units: list[dict[str, Any]]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for left, right in zip(strokes, strokes[1:]):
        edges.append({"type": "temporal_next", "source": f"shot_{left['shot_id']}", "target": f"shot_{right['shot_id']}"})
    by_player: dict[str, list[int]] = {}
    for shot in strokes:
        player = (shot.get("parsed") or {}).get("hitter")
        if player in {None, "", "-", "unknown", "Unknown"}:
            continue
        player = str(player)
        by_player.setdefault(player, []).append(int(shot["shot_id"]))
    for ids in by_player.values():
        for left, right in zip(ids, ids[1:]):
            edges.append({"type": "same_player_next", "source": f"shot_{left}", "target": f"shot_{right}"})
    for unit in units:
        uid = unit["unit_id"]
        for sid in unit.get("setup_shots") or []:
            edges.append({"type": "setup_to_unit", "source": f"shot_{sid}", "target": uid})
        for sid in unit.get("key_action_shots") or []:
            edges.append({"type": "unit_to_key", "source": uid, "target": f"shot_{sid}"})
        for sid in unit.get("followup_shots") or []:
            edges.append({"type": "key_to_followup", "source": uid, "target": f"shot_{sid}"})
    return edges


def build_graph(record: dict[str, Any], split: str) -> dict[str, Any]:
    clip = str(record.get("clip_id") or record.get("video"))
    start, end = clip_frame_range(clip)
    strokes = build_strokes(record)
    units = [build_unit(unit, idx, strokes) for idx, unit in enumerate(record.get("tactical_units") or [], start=1)]
    return {
        "rally_id": clip,
        "clip_id": clip,
        "video": clip,
        "match_id": record.get("match_id") or match_id_from_clip(clip),
        "split": split,
        "fps": record.get("fps"),
        "height": record.get("height"),
        "width": record.get("width"),
        "num_frames": record.get("num_frames"),
        "clip_frame_start": start,
        "clip_frame_end": end,
        "media": record.get("media")
        or {
            "video_path": f"media/videos/{match_id_from_clip(clip)}.mp4",
            "frames_dir": f"media/frames_224/{clip}",
        },
        "players": {
            "far_name": record.get("far_name"),
            "far_hand": record.get("far_hand"),
            "near_name": record.get("near_name"),
            "near_hand": record.get("near_hand"),
        },
        "score": {
            "far_set": record.get("far_set"),
            "far_game": record.get("far_game"),
            "far_point": record.get("far_point"),
            "near_set": record.get("near_set"),
            "near_game": record.get("near_game"),
            "near_point": record.get("near_point"),
        },
        "rally_summary": record.get("rally_summary") or {},
        "strokes": strokes,
        "tactical_units": units,
        "primary_tactical_units": [u["unit_id"] for u in units if u.get("is_primary_eval")],
        "edges": build_edges(strokes, units),
    }


def validate_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    strokes = graph.get("strokes") or []
    n = len(strokes)
    node_ids = {f"shot_{s['shot_id']}" for s in strokes}
    node_ids.update({u["unit_id"] for u in graph.get("tactical_units") or []})
    frames_by_id = {s["shot_id"]: s["frame"] for s in strokes}
    for shot in strokes:
        parsed = shot.get("parsed") or {}
        if not parsed.get("parse_ok"):
            errors.append(f"{graph.get('rally_id')}: label_parse_failed shot={shot.get('shot_id')} label={shot.get('label')}")
    for unit in graph.get("tactical_units") or []:
        ev = unit.get("gold_evidence") or {}
        shot_ids = ev.get("shot_ids") or []
        frames = ev.get("frames") or []
        labels = ev.get("event_labels") or []
        for sid in shot_ids:
            if not isinstance(sid, int) or sid < 1 or sid > n:
                errors.append(f"{graph.get('rally_id')} {unit.get('unit_id')}: bad evidence shot_id={sid}")
        expected = len([sid for sid in shot_ids if sid in frames_by_id])
        if len(frames) != expected:
            errors.append(f"{graph.get('rally_id')} {unit.get('unit_id')}: frame/evidence length mismatch")
        if len(labels) != expected:
            errors.append(f"{graph.get('rally_id')} {unit.get('unit_id')}: label/evidence length mismatch")
    for edge in graph.get("edges") or []:
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            errors.append(f"{graph.get('rally_id')}: bad edge {edge}")
    return errors


def graph_report(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    primary_counts: Counter[str] = Counter()
    for graph in graphs:
        for unit in graph.get("tactical_units") or []:
            if unit.get("is_primary_eval"):
                primary_counts[str(unit.get("level_3"))] += 1
    return {"primary_level3_counts": dict(primary_counts.most_common())}
