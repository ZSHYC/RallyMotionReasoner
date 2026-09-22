from __future__ import annotations

import json
from typing import Any

from tennisvar.schema import OPTIONAL_SCHEMA, REQUIRED_SCHEMA, validate_prediction_payload


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
        if not isinstance(payload, dict) or not set(REQUIRED_SCHEMA).issubset(payload) or not set(payload).issubset(allowed):
            raise ValueError(f"Qwen SFT assistant schema mismatch: {row.get('qa_id')}")
        violations = validate_prediction_payload(payload)
        if violations:
            raise ValueError(f"Qwen SFT assistant values violate schema for {row.get('qa_id')}: {violations}")
        messages.append({"role": "assistant", "content": answer})
    return messages


def prepare_qwen_sft_item(processor: Any, row: dict[str, Any], device: Any) -> dict[str, Any]:
    """Create a batch of one and mask every token except the assistant answer."""
    import torch
    from qwen_vl_utils import process_vision_info

    full_messages = structured_video_messages(row, include_answer=True)
    prompt_messages = structured_video_messages(row, include_answer=False)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(full_messages, return_video_kwargs=True)
    inputs = processor(text=[full_text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs
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
