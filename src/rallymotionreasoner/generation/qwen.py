from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rallymotionreasoner.schema import REQUIRED_SCHEMA, validate_prediction_payload

REQUIRED_QWEN_FIELDS = set(REQUIRED_SCHEMA)


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
    allowed = [int(item["shot_id"]) for item in candidates]
    return (
        "You are a tennis video evidence QA system. Use only the video and PREDICTED event candidates below. "
        "Candidates can be wrong. Do not treat them as ground truth and do not invent shot IDs.\n"
        f"Question: {question}\n"
        f"Predicted candidates: {json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}\n"
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
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
        self.torch = torch
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = ModelClass.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
        ).eval()
        self.adapter = None
        self.adapter_manifest = None
        if adapter:
            from peft import PeftModel

            from rallymotionreasoner.generation.manifest import load_qwen_adapter_manifest

            adapter = Path(adapter)
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"adapter_config.json not found: {adapter}")
            self.adapter_manifest = load_qwen_adapter_manifest(
                adapter,
                expected_track="graph_reasoner_assisted_pred",
            )
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False).eval()
            self.adapter = str(adapter)

    def generate(self, frame_paths: list[Path], question: str, candidates: list[dict[str, Any]], *, max_new_tokens: int = 512) -> dict[str, Any]:
        if not frame_paths:
            raise ValueError("Qwen video inference requires non-empty frames")
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils is required; image-mode fallback is intentionally disabled") from exc
        prompt = qwen_prompt(question, candidates)
        messages = [{"role": "user", "content": [{"type": "video", "video": [str(path) for path in frame_paths]}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt", **video_kwargs)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)[0]
        parsed = parse_qwen_json(raw)
        allowed = {int(item["shot_id"]) for item in candidates}
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
        for field in ("evidence_shot_ids", "key_action_shot_ids"):
            values = parsed.get(field) if isinstance(parsed.get(field), list) else []
            parsed[field] = list(dict.fromkeys(int(value) for value in values if str(value).isdigit() and int(value) in allowed))
        parsed["key_action_shot_ids"] = [sid for sid in parsed["key_action_shot_ids"] if sid in parsed["evidence_shot_ids"]]
        by_id = {int(item["shot_id"]): item for item in candidates}
        parsed["evidence_frames"] = [int(by_id[sid]["frame"]) for sid in parsed["evidence_shot_ids"]]
        parsed["raw"] = raw
        return parsed
