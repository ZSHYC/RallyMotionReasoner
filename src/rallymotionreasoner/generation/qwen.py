from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from rallymotionreasoner.schema import REQUIRED_SCHEMA, validate_prediction_payload

REQUIRED_QWEN_FIELDS = set(REQUIRED_SCHEMA)
QWEN_SAMPLING_CONTRACT = {
    "max_frames": 32,
    "global_frames": 16,
    "local_frames": 4,
    "local_radius": 8,
}
QWEN_PROMPT_HEADER = (
    "You are a tennis video evidence QA system. Use only the video and PREDICTED event candidates below. "
    "Candidates can be wrong. Do not treat them as ground truth and do not invent shot IDs.\nQuestion: "
)
QWEN_CANDIDATES_MARKER = "\nPredicted candidates: "
QWEN_GENERATION_CONTRACT = {
    "prompt": "rallymotionreasoner.qwen_prompt.v1",
    "sampling": QWEN_SAMPLING_CONTRACT,
    "source_timing": {
        "frame_indices": "e2e_metadata.video_frame_indices",
        "fps": "e2e_metadata.video_fps",
        "required": True,
    },
}


def _validate_video_timing(
    frame_count: int,
    frame_indices: list[int] | None,
    fps: float | None,
    *,
    required: bool = False,
) -> tuple[list[int] | None, float | None]:
    if frame_indices is None and fps is None:
        if required:
            raise ValueError("video_frame_indices and video_fps are required for this Qwen row")
        return None, None
    if frame_indices is None or fps is None:
        raise ValueError("frame_indices and fps must be provided together")
    if len(frame_indices) != frame_count:
        raise ValueError("frame_indices and video frames must have the same count")
    if not frame_indices or any(not isinstance(value, int) or isinstance(value, bool) for value in frame_indices):
        raise ValueError("frame_indices must be a non-empty integer list")
    if frame_indices[0] < 0 or any(right <= left for left, right in zip(frame_indices, frame_indices[1:])):
        raise ValueError("frame_indices must be non-negative and strictly increasing")
    if isinstance(fps, bool):
        raise ValueError("fps must be a positive finite number")
    try:
        normalized_fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a positive finite number") from exc
    if not math.isfinite(normalized_fps) or normalized_fps <= 0:
        raise ValueError("fps must be a positive finite number")
    return list(frame_indices), normalized_fps


def _prepare_qwen3_vision_inputs(
    messages: list[dict[str, Any]],
    *,
    frame_count: int,
    frame_indices: list[int] | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    frame_indices, fps = _validate_video_timing(frame_count, frame_indices, fps)

    from qwen_vl_utils import process_vision_info

    images, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if video_inputs is None:
        videos, video_metadata = None, None
    else:
        videos, metadata = zip(*video_inputs)
        videos, video_metadata = list(videos), [dict(item) for item in metadata]

    if frame_indices is not None:
        if video_metadata is None or len(video_metadata) != 1:
            raise ValueError("source timing requires exactly one processed video")
        processed_indices = video_metadata[0].get("frames_indices")
        if not isinstance(processed_indices, list | tuple) or len(processed_indices) < len(frame_indices):
            raise ValueError("processed video frame count does not match source timing")
        video_metadata[0]["frames_indices"] = frame_indices + [frame_indices[-1]] * (
            len(processed_indices) - len(frame_indices)
        )
        video_metadata[0]["fps"] = float(fps)
        video_metadata[0]["total_num_frames"] = frame_indices[-1] + 1
    return {
        "images": images,
        "videos": videos,
        "video_metadata": video_metadata,
        "return_tensors": "pt",
        **video_kwargs,
        "do_resize": False,
    }


def _ground_qwen_evidence(
    prediction: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    grounded = dict(prediction)
    allowed = {int(item["shot_id"]) for item in candidates}
    for field in ("evidence_shot_ids", "key_action_shot_ids"):
        values = grounded.get(field) if isinstance(grounded.get(field), list) else []
        grounded[field] = list(
            dict.fromkeys(int(value) for value in values if str(value).isdigit() and int(value) in allowed)
        )
    grounded["key_action_shot_ids"] = [
        shot_id for shot_id in grounded["key_action_shot_ids"] if shot_id in grounded["evidence_shot_ids"]
    ]
    by_id = {int(item["shot_id"]): item for item in candidates}
    grounded["evidence_frames"] = [int(by_id[shot_id]["frame"]) for shot_id in grounded["evidence_shot_ids"]]
    if grounded.get("answerability") != "unanswerable" and not grounded["evidence_shot_ids"]:
        grounded.update(
            answer="",
            level_1=None,
            level_2=None,
            level_3=None,
            observed_effect="unknown",
            causal_strength="insufficient",
            answerability="unanswerable",
            explanation="The generated answer did not cite a valid evidence candidate.",
            generation_error="invalid_evidence",
        )
    return grounded


def parse_qwen_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not validate_prediction_payload(value):
            return value
    return None


def qwen_prompt(question: str, candidates: list[dict[str, Any]]) -> str:
    if not str(question).strip() or not candidates:
        raise ValueError("Qwen prompt requires a non-empty question and candidate list")
    allowed = [item.get("shot_id") for item in candidates if isinstance(item, dict)]
    if (
        len(allowed) != len(candidates)
        or any(not isinstance(shot_id, int) or isinstance(shot_id, bool) for shot_id in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise ValueError("Qwen candidates require a unique integer shot_id")
    return (
        f"{QWEN_PROMPT_HEADER}{question}"
        f"{QWEN_CANDIDATES_MARKER}{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}\n"
        "Return exactly one JSON object with fields: answer (string), answer_type "
        "(short_answer|free_form|verification|counterfactual_limited), level_1/level_2/level_3 "
        "(string or null), evidence_shot_ids/key_action_shot_ids/evidence_frames (integer arrays), "
        "observed_effect (winner|forced_error|unforced_error|continuation|countered|unknown), causal_strength "
        "(verified|plausible_rule_based|not_verified|insufficient), answerability "
        "(answerable|partially_answerable|unanswerable), and explanation (string). "
        f"Every evidence ID must belong to {allowed}. Use cautious wording for causal claims."
    )


class QwenVideoBackend:
    def __init__(self, model_path: Path, *, adapter: Path | None = None, device_map: str | None = "auto") -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        self.adapter = None
        self.adapter_manifest = None
        if adapter:
            from rallymotionreasoner.generation.manifest import load_qwen_adapter_manifest

            adapter = Path(adapter)
            self.adapter_manifest = load_qwen_adapter_manifest(
                adapter,
                expected_track="graph_reasoner_assisted_pred",
                expected_base_model=self.model_path,
            )
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = ModelClass.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
        ).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False).eval()
            self.adapter = str(adapter)

    def generate(
        self,
        frame_paths: list[Path],
        question: str,
        candidates: list[dict[str, Any]],
        *,
        max_new_tokens: int = 512,
        frame_indices: list[int] | None = None,
        fps: float | None = None,
    ) -> dict[str, Any]:
        if not frame_paths:
            raise ValueError("Qwen video inference requires non-empty frames")
        if self.adapter_manifest is not None:
            _validate_video_timing(len(frame_paths), frame_indices, fps, required=True)
        prompt = qwen_prompt(question, candidates)
        messages = [{"role": "user", "content": [{"type": "video", "video": [str(path) for path in frame_paths]}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            processor_kwargs = _prepare_qwen3_vision_inputs(
                messages,
                frame_count=len(frame_paths),
                frame_indices=frame_indices,
                fps=fps,
            )
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils is required; image-mode fallback is intentionally disabled") from exc
        inputs = self.processor(text=[text], **processor_kwargs)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)[0]
        parsed = parse_qwen_json(raw)
        if parsed is None:
            return {
                "answer": "",
                "answer_type": "free_form",
                "level_1": None,
                "level_2": None,
                "level_3": None,
                "answerability": "unanswerable",
                "explanation": "Qwen output did not satisfy the required JSON schema.",
                "evidence_shot_ids": [],
                "key_action_shot_ids": [],
                "evidence_frames": [],
                "observed_effect": "unknown",
                "causal_strength": "insufficient",
                "parse_error": "invalid_qwen_json",
                "raw": raw,
            }
        parsed = _ground_qwen_evidence(parsed, candidates)
        parsed["raw"] = raw
        return parsed
