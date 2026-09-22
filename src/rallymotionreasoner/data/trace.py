from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rallymotionreasoner.graph import build_graph, match_id_from_clip, validate_graph
from rallymotionreasoner.io import write_json, write_jsonl
from rallymotionreasoner.schema import normalize_effect
from rallymotionreasoner.video import list_images

DATASET_NAME = "trace"
SPLITS = ("train", "val", "test")


def _ints(values: Any) -> list[int]:
    out: list[int] = []
    for value in values if isinstance(values, list) else []:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item not in out:
            out.append(item)
    return out


def reasoning_evidence(qa: dict[str, Any]) -> tuple[list[int], list[int], str]:
    evidence: list[int] = []
    key: list[int] = []
    explanations: list[str] = []
    for item in (qa.get("reasoning_chain") or {}).get("evidence") or []:
        if not isinstance(item, dict):
            continue
        evidence.extend(_ints(item.get("evidence_shots")))
        key.extend(_ints(item.get("key_action_shots")))
        text = str(item.get("explanation") or "").strip()
        if text:
            explanations.append(text)
    return _ints(evidence), _ints(key), " ".join(explanations)


def _chain_ids(node: Any) -> list[int]:
    if isinstance(node, dict):
        out: list[int] = []
        for key, value in node.items():
            out.extend(_ints(value) if key == "shot_ids" else _chain_ids(value))
        return _ints(out)
    if isinstance(node, list):
        out = []
        for value in node:
            out.extend(_chain_ids(value))
        return _ints(out)
    return []


def _choose_tactical_unit(row: dict[str, Any], evidence: list[int], key: list[int]) -> dict[str, Any]:
    units = [item for item in row.get("tactical_units") or [] if isinstance(item, dict)]
    if not units:
        return {}
    evidence_set, key_set = set(evidence), set(key)

    def score(item: dict[str, Any]) -> tuple[int, int]:
        chain = item.get("evidence_chain") or {}
        unit_ids = set(_chain_ids(chain))
        key_ids = set(_chain_ids(chain.get("key_action") if isinstance(chain, dict) else {}))
        return len(key_set & key_ids), len(evidence_set & unit_ids)

    return max(units, key=score)


def _causal_strength(result: Any) -> str:
    value = str(result or "").lower().replace("-", "_")
    if value in {"key_action_winner", "opponent_forced_error"}:
        return "verified"
    if value in {"execution_forced_error", "induced_forced_error"}:
        return "plausible_rule_based"
    if value:
        return "not_verified"
    return "insufficient"


def build_reference(
    row: dict[str, Any], split: str, qa: dict[str, Any], graph: dict[str, Any], *, qa_index: int
) -> dict[str, Any]:
    video = str(row["video"])
    evidence, key, explanation = reasoning_evidence(qa)
    unit = _choose_tactical_unit(row, evidence, key)
    frame_by_shot = {int(shot["shot_id"]): int(shot["frame"]) for shot in graph.get("strokes") or []}
    qa_group = str(qa.get("qa_group") or "UNKNOWN")
    question = str(qa.get("question") or "").strip()
    answerability = "answerable" if question and str(qa.get("answer") or "").strip() else "unanswerable"
    result = unit.get("tactic_result")
    gold = {
        "answer": str(qa.get("answer") or ""),
        "answer_type": "short_answer" if qa_group == "Q1" else "free_form",
        "level_1": unit.get("level_1"),
        "level_2": unit.get("level_2"),
        "level_3": unit.get("level_3"),
        "evidence_shot_ids": evidence,
        "key_action_shot_ids": key,
        "evidence_frames": [frame_by_shot[item] for item in evidence if item in frame_by_shot],
        "observed_effect": normalize_effect(result),
        "causal_strength": _causal_strength(result),
        "answerability": answerability,
        "explanation": explanation,
    }
    return {
        "schema_version": "rallymotionreasoner.qa.v3",
        "dataset": DATASET_NAME,
        "split": split,
        "qa_id": f"{split}:{video}:{qa_index}",
        "rally_id": video,
        "match_id": match_id_from_clip(video),
        "qa_group": qa_group,
        "qa_type": qa_group,
        "question": question,
        "gold_answer": gold,
        # Alignment-only supervision. These fields are intentionally outside
        # gold_answer so they can never enter a Qwen assistant target.
        "key_action_frames": [frame_by_shot[item] for item in key if item in frame_by_shot],
        "gold_shot_frames": {str(shot_id): frame for shot_id, frame in sorted(frame_by_shot.items())},
    }


def validate_source_row(row: dict[str, Any], split: str, index: int) -> list[str]:
    prefix = f"{split}[{index}]"
    issues: list[str] = []
    video = str(row.get("video") or "")
    if not video:
        issues.append(f"{prefix}: missing video")
    events = row.get("events")
    if not isinstance(events, list) or not events:
        issues.append(f"{prefix}/{video}: events must be a non-empty list")
        return issues
    frames: list[int] = []
    for event_index, event in enumerate(events, start=1):
        try:
            frame = int(event.get("frame"))
        except (AttributeError, TypeError, ValueError):
            issues.append(f"{prefix}/{video}: invalid event frame at {event_index}")
            continue
        frames.append(frame)
        if frame < 0 or frame >= int(row.get("num_frames") or 0):
            issues.append(f"{prefix}/{video}: event frame {frame} out of range")
        if not str(event.get("label") or ""):
            issues.append(f"{prefix}/{video}: empty event label at {event_index}")
    if frames != sorted(frames):
        issues.append(f"{prefix}/{video}: events are not time ordered")
    qas = row.get("qa_units")
    if not isinstance(qas, list) or not qas:
        issues.append(f"{prefix}/{video}: qa_units must be non-empty")
    valid_shots = set(range(1, len(events) + 1))
    for qa_index, qa in enumerate(qas or [], start=1):
        if not str(qa.get("question") or "").strip() or not str(qa.get("answer") or "").strip():
            issues.append(f"{prefix}/{video}: empty question or answer at qa {qa_index}")
        evidence, key, _ = reasoning_evidence(qa)
        invalid = sorted((set(evidence) | set(key)) - valid_shots)
        if invalid:
            issues.append(f"{prefix}/{video}: invalid evidence shot ids {invalid}")
    return issues


def load_source(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"source must be a JSON array of objects: {path}")
    return payload


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    frame_root: Path,
    *,
    strict_media: bool = True,
) -> dict[str, Any]:
    source_root, output_root, frame_root = Path(source_root), Path(output_root), Path(frame_root)
    split_videos: dict[str, set[str]] = {}
    all_issues: list[str] = []
    split_reports: dict[str, Any] = {}
    qa_groups: Counter[str] = Counter()
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source = source_root / f"{split}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = load_source(source)
        split_videos[split] = {str(row.get("video")) for row in rows}
        graphs: list[dict[str, Any]] = []
        refs: list[dict[str, Any]] = []
        media: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            all_issues.extend(validate_source_row(row, split, index))
            video = str(row.get("video"))
            frame_dir = frame_root / video
            images = list_images(frame_dir)
            media.append(
                {
                    "rally_id": video,
                    "split": split,
                    "frames_dir": str(frame_dir),
                    "exists": frame_dir.is_dir(),
                    "frame_count": len(images),
                    "expected_num_frames": int(row.get("num_frames") or 0),
                    "fps": float(row.get("fps") or 25.0),
                    "width": int(row.get("width") or 0),
                    "height": int(row.get("height") or 0),
                }
            )
            normalized = dict(row)
            normalized["clip_id"] = video
            normalized["media"] = {"frames_dir": str(frame_dir)}
            graph = build_graph(normalized, split)
            all_issues.extend(f"{split}/{video}: {error}" for error in validate_graph(graph))
            graphs.append(graph)
            for qa_index, qa in enumerate(row.get("qa_units") or [], start=1):
                ref = build_reference(row, split, qa, graph, qa_index=qa_index)
                refs.append(ref)
                qa_groups[ref["qa_group"]] += 1
        write_jsonl(output_root / "graphs" / f"sgtr_graph_{DATASET_NAME}_{split}.jsonl", graphs)
        write_jsonl(output_root / "e2e_refs" / f"{DATASET_NAME}_e2e_{split}.jsonl", refs)
        write_jsonl(output_root / "manifests" / f"media_{split}.jsonl", media)
        missing = [item for item in media if not item["exists"] or item["frame_count"] == 0]
        mismatched = [item for item in media if item["frame_count"] != item["expected_num_frames"]]
        split_reports[split] = {
            "source": str(source),
            "rows": len(rows),
            "qa_rows": len(refs),
            "events": sum(len(row.get("events") or []) for row in rows),
            "media_present": len(media) - len(missing),
            "media_missing": len(missing),
            "frame_count_mismatch": len(mismatched),
            "missing_sample": missing[:20],
            "mismatch_sample": mismatched[:20],
        }
    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = sorted(split_videos[left] & split_videos[right])
    report = {
        "status": "PASS",
        "dataset": DATASET_NAME,
        "schema_version": "rallymotionreasoner.dataset.v1",
        "splits": split_reports,
        "qa_groups": dict(sorted(qa_groups.items())),
        "video_overlap": {key: len(value) for key, value in overlaps.items()},
        "video_overlap_sample": {key: value[:20] for key, value in overlaps.items()},
        "validation_issue_count": len(all_issues),
        "validation_issue_sample": all_issues[:100],
    }
    if all_issues or any(overlaps.values()):
        report["status"] = "FAIL_SCHEMA_OR_SPLIT"
    if strict_media and any(item["media_missing"] for item in split_reports.values()):
        report["status"] = "FAIL_MISSING_MEDIA"
    write_json(output_root / "manifests" / "dataset_audit.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"dataset preparation failed: {report['status']}; see {output_root / 'manifests/dataset_audit.json'}")
    return report


def iter_event_labels(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    for row in rows:
        for event in row.get("events") or []:
            label = str(event.get("label") or "")
            if label:
                yield label
