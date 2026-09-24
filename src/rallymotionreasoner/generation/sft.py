from __future__ import annotations

import json
from typing import Any

from rallymotionreasoner.generation.qwen import (
    QWEN_CANDIDATES_MARKER,
    QWEN_PROMPT_HEADER,
    QWEN_SAMPLING_CONTRACT,
    _prepare_qwen3_vision_inputs,
    _validate_video_timing,
    qwen_prompt,
)
from rallymotionreasoner.schema import OPTIONAL_SCHEMA, REQUIRED_SCHEMA, validate_prediction_payload


def _sft_video_timing(
    row: dict[str, Any], *, frame_count: int | None = None, required: bool = False
) -> tuple[list[int] | None, float | None]:
    metadata = row.get("e2e_metadata") or {}
    frame_indices = metadata.get("video_frame_indices")
    fps = metadata.get("video_fps")
    try:
        return _validate_video_timing(frame_count or len(frame_indices or []), frame_indices, fps, required=required)
    except ValueError as exc:
        raise ValueError(f"e2e_metadata.video_frame_indices and video_fps: {exc}") from exc


def validate_sft_generation_contract(row: dict[str, Any]) -> None:
    source = row.get("messages") or []
    if any(item.get("role") == "system" for item in source):
        raise ValueError(f"Qwen SFT system messages are incompatible with online generation: {row.get('qa_id')}")
    messages = structured_video_messages(row, include_answer=True)
    frames = row["videos"][0]
    if len(frames) > QWEN_SAMPLING_CONTRACT["max_frames"]:
        raise ValueError(
            f"Qwen SFT row exceeds the {QWEN_SAMPLING_CONTRACT['max_frames']}-frame contract: {row.get('qa_id')}"
        )
    frame_indices, _ = _sft_video_timing(row, frame_count=len(frames), required=True)
    user_message = next(item for item in messages if item["role"] == "user")
    user = str(user_message["content"][-1]["text"])
    if not user.startswith(QWEN_PROMPT_HEADER) or QWEN_CANDIDATES_MARKER not in user:
        raise ValueError(f"Qwen SFT prompt contract mismatch: {row.get('qa_id')}")
    question, _, tail = user[len(QWEN_PROMPT_HEADER) :].partition(QWEN_CANDIDATES_MARKER)
    candidate_json, separator, _ = tail.partition("\nReturn exactly one JSON object")
    try:
        candidates = json.loads(candidate_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen SFT prompt contract mismatch: {row.get('qa_id')}") from exc
    valid_candidates = (
        isinstance(candidates, list)
        and candidates
        and all(
            isinstance(candidate, dict)
            and isinstance(candidate.get("frame"), int)
            and not isinstance(candidate.get("frame"), bool)
            for candidate in candidates
        )
    )
    try:
        canonical = qwen_prompt(question, candidates) if valid_candidates else None
    except ValueError:
        canonical = None
    if not separator or canonical is None or user != canonical:
        raise ValueError(f"Qwen SFT prompt contract mismatch: {row.get('qa_id')}")
    sampled_frames = set(frame_indices or [])
    if any(int(candidate["frame"]) not in sampled_frames for candidate in candidates):
        raise ValueError(f"Qwen SFT candidate frames must occur in video_frame_indices: {row.get('qa_id')}")
    target = json.loads(str(messages[-1]["content"]))
    candidate_frames = {int(candidate["shot_id"]): int(candidate["frame"]) for candidate in candidates}
    evidence_ids = target["evidence_shot_ids"]
    if any(shot_id not in candidate_frames for shot_id in evidence_ids):
        raise ValueError(f"Qwen SFT evidence IDs must occur in prompt candidates: {row.get('qa_id')}")
    if target["evidence_frames"] != [candidate_frames[shot_id] for shot_id in evidence_ids]:
        raise ValueError(f"Qwen SFT evidence frames must match prompt candidates: {row.get('qa_id')}")
    if target["answerability"] != "unanswerable" and not evidence_ids:
        raise ValueError(f"Qwen SFT answerable targets require prompt evidence: {row.get('qa_id')}")
    if not evidence_ids and (
        target["answer"] != ""
        or any(target[level] is not None for level in ("level_1", "level_2", "level_3"))
        or target["observed_effect"] != "unknown"
        or target["causal_strength"] != "insufficient"
    ):
        raise ValueError(f"Qwen SFT targets without evidence must match online abstention: {row.get('qa_id')}")


def structured_video_messages(row: dict[str, Any], *, include_answer: bool) -> list[dict[str, Any]]:
    """Convert one immutable training row into native Qwen3-VL messages."""
    source = row.get("messages") or []
    system = next((item.get("content") for item in source if item.get("role") == "system"), None)
    user = next((str(item.get("content") or "") for item in source if item.get("role") == "user"), "")
    answer = next((str(item.get("content") or "") for item in source if item.get("role") == "assistant"), "")
    videos = row.get("videos") or []
    if len(videos) != 1 or not isinstance(videos[0], list) or not videos[0]:
        raise ValueError(f"Qwen SFT row requires exactly one non-empty frame-list video: {row.get('qa_id')}")
    if user.startswith("<video>\n"):
        user = user[len("<video>\n") :]
    if not user.strip():
        raise ValueError(f"Qwen SFT row has an empty user prompt: {row.get('qa_id')}")
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "video", "video": [str(path) for path in videos[0]]},
                {"type": "text", "text": user},
            ],
        }
    )
    if include_answer:
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Qwen SFT assistant target is not JSON: {row.get('qa_id')}") from exc
        allowed = set(REQUIRED_SCHEMA) | set(OPTIONAL_SCHEMA)
        if (
            not isinstance(payload, dict)
            or not set(REQUIRED_SCHEMA).issubset(payload)
            or not set(payload).issubset(allowed)
        ):
            raise ValueError(f"Qwen SFT assistant schema mismatch: {row.get('qa_id')}")
        violations = validate_prediction_payload(payload)
        if violations:
            raise ValueError(f"Qwen SFT assistant values violate schema for {row.get('qa_id')}: {violations}")
        messages.append({"role": "assistant", "content": answer})
    return messages


def prepare_qwen_sft_item(processor: Any, row: dict[str, Any], device: Any) -> dict[str, Any]:
    """Create a batch of one and mask every token except the assistant answer."""
    import torch

    full_messages = structured_video_messages(row, include_answer=True)
    prompt_messages = structured_video_messages(row, include_answer=False)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    frame_indices, fps = _sft_video_timing(row, frame_count=len(row["videos"][0]))
    processor_kwargs = _prepare_qwen3_vision_inputs(
        full_messages,
        frame_count=len(row["videos"][0]),
        frame_indices=frame_indices,
        fps=fps,
    )
    inputs = processor(text=[full_text], **processor_kwargs)
    prompt_inputs = processor(
        text=[prompt_text],
        **processor_kwargs,
    )
    prompt_length = int(prompt_inputs["attention_mask"][0].sum())
    full_ids = inputs["input_ids"][0]
    prompt_ids = prompt_inputs["input_ids"][0, :prompt_length]
    if full_ids.shape[0] <= prompt_length or not torch.equal(full_ids[:prompt_length], prompt_ids):
        raise RuntimeError(f"Qwen prompt tokens are not an exact prefix of the SFT row: {row.get('qa_id')}")
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    labels[inputs["attention_mask"] == 0] = -100
    if not bool((labels != -100).any()):
        raise RuntimeError(f"Qwen SFT row has no supervised assistant tokens: {row.get('qa_id')}")
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    batch["labels"] = labels.to(device)
    return batch
