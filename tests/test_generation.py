import json
import sys
from types import SimpleNamespace

import pytest

from rallymotionreasoner.generation.qwen import _prepare_qwen3_vision_inputs, parse_qwen_json, qwen_prompt
from rallymotionreasoner.generation.sft import _sft_video_timing


def _prediction() -> dict:
    return {
        "answer": "Player Near created space with the earlier strokes.",
        "answer_type": "free_form",
        "level_1": "B",
        "level_2": "B2",
        "level_3": "net_attack",
        "evidence_shot_ids": [1, 3],
        "key_action_shot_ids": [3],
        "evidence_frames": [12, 36],
        "observed_effect": "continuation",
        "causal_strength": "verified",
        "answerability": "answerable",
        "explanation": "The selected strokes support the tactical conclusion.",
    }


def test_qwen_json_parser_accepts_schema_only() -> None:
    payload = _prediction()
    assert parse_qwen_json(f"```json\n{json.dumps(payload)}\n```") == payload
    del payload["evidence_frames"]
    assert parse_qwen_json(json.dumps(payload)) is None


def test_prompt_limits_evidence_ids() -> None:
    prompt = qwen_prompt("What changed?", [{"shot_id": 2}, {"shot_id": 5}])
    assert "[2, 5]" in prompt
    assert "do not invent shot IDs" in prompt


def test_qwen3_vision_inputs_preserve_original_frame_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def process_vision_info(messages, **kwargs):
        calls.append((messages, kwargs))
        metadata = {"fps": 2.0, "frames_indices": [0, 1, 2, 3], "total_num_frames": 4}
        return None, [("video", metadata)], {"do_sample_frames": False}

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process_vision_info))
    messages = [{"role": "user", "content": [{"type": "video", "video": ["a", "b", "c"]}]}]

    processor_kwargs = _prepare_qwen3_vision_inputs(
        messages,
        frame_count=3,
        frame_indices=[4, 11, 25],
        fps=30.0,
    )

    assert processor_kwargs == {
        "images": None,
        "videos": ["video"],
        "video_metadata": [{"fps": 30.0, "frames_indices": [4, 11, 25, 25], "total_num_frames": 26}],
        "return_tensors": "pt",
        "do_sample_frames": False,
        "do_resize": False,
    }
    assert calls[0][1] == {
        "image_patch_size": 16,
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }


def test_qwen3_vision_inputs_keep_default_timing_without_source_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    default_metadata = {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2}
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(
            process_vision_info=lambda *args, **kwargs: (None, [("video", default_metadata)], {})
        ),
    )

    processor_kwargs = _prepare_qwen3_vision_inputs([], frame_count=2)

    assert processor_kwargs["video_metadata"] == [default_metadata]


def test_qwen3_vision_inputs_reject_bad_source_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(
            process_vision_info=lambda *args, **kwargs: (
                None,
                [("video", {"fps": 2.0, "frames_indices": [0, 1, 2]})],
                {},
            )
        ),
    )
    messages = [{"role": "user", "content": [{"type": "video", "video": ["a", "b", "c"]}]}]

    with pytest.raises(ValueError, match="strictly increasing"):
        _prepare_qwen3_vision_inputs(
            messages,
            frame_count=3,
            frame_indices=[4, 4, 25],
            fps=30.0,
        )
    with pytest.raises(ValueError, match="provided together"):
        _prepare_qwen3_vision_inputs(messages, frame_count=3, frame_indices=[4, 11, 25])
    with pytest.raises(ValueError, match="same count"):
        _prepare_qwen3_vision_inputs(messages, frame_count=2, frame_indices=[4, 11, 25], fps=30.0)


def test_sft_video_timing_uses_e2e_metadata_only_when_complete() -> None:
    row = {"e2e_metadata": {"video_frame_indices": [4, 11, 25], "video_fps": 30.0}}
    assert _sft_video_timing(row) == ([4, 11, 25], 30.0)
    assert _sft_video_timing({"e2e_metadata": {}}) == (None, None)

    with pytest.raises(ValueError, match="provided together"):
        _sft_video_timing({"e2e_metadata": {"video_fps": 30.0}})
