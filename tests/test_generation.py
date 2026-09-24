import json
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rallymotionreasoner.generation.manifest import (
    MANIFEST_NAME,
    QWEN_ADAPTER_SCHEMA,
    load_qwen_adapter_manifest,
)
from rallymotionreasoner.generation.qwen import (
    QWEN_GENERATION_CONTRACT,
    QWEN_SAMPLING_CONTRACT,
    QwenVideoBackend,
    _ground_qwen_evidence,
    _prepare_qwen3_vision_inputs,
    parse_qwen_json,
    qwen_prompt,
)
from rallymotionreasoner.generation.sft import _sft_video_timing, validate_sft_generation_contract


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

    with pytest.raises(ValueError, match="non-empty"):
        qwen_prompt("What changed?", [])
    with pytest.raises(ValueError, match="unique integer shot_id"):
        qwen_prompt("What changed?", [{"shot_id": 2}, {"shot_id": 2}])


def test_qwen_sampling_contract_contains_only_executed_frame_policy() -> None:
    assert QWEN_SAMPLING_CONTRACT == {
        "max_frames": 32,
        "global_frames": 16,
        "local_frames": 4,
        "local_radius": 8,
    }


def test_invalid_qwen_evidence_forces_abstention() -> None:
    payload = _prediction()
    payload["evidence_shot_ids"] = [999]
    payload["key_action_shot_ids"] = [999]

    grounded = _ground_qwen_evidence(payload, [{"shot_id": 2, "frame": 12}])

    assert grounded["answer"] == ""
    assert grounded["answerability"] == "unanswerable"
    assert grounded["evidence_shot_ids"] == []
    assert grounded["key_action_shot_ids"] == []
    assert grounded["evidence_frames"] == []
    assert grounded["causal_strength"] == "insufficient"
    assert grounded["generation_error"] == "invalid_evidence"


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


def test_new_sft_rows_match_inference_prompt_and_timing_contract() -> None:
    candidates = [{"shot_id": 2, "frame": 12}]
    row = {
        "qa_id": "qa-1",
        "videos": [["a.jpg", "b.jpg"]],
        "messages": [{"role": "user", "content": qwen_prompt("What changed?", candidates)}],
        "e2e_metadata": {"video_frame_indices": [4, 12], "video_fps": 30.0},
    }

    validate_sft_generation_contract(row)

    row["messages"][0]["content"] = "A different training prompt"
    with pytest.raises(ValueError, match="prompt contract"):
        validate_sft_generation_contract(row)

    row["messages"][0]["content"] = qwen_prompt("What changed?", candidates)
    del row["e2e_metadata"]["video_fps"]
    with pytest.raises(ValueError, match="provided together"):
        validate_sft_generation_contract(row)

    row["e2e_metadata"] = {"video_frame_indices": [4, 11], "video_fps": 30.0}
    with pytest.raises(ValueError, match="candidate frames"):
        validate_sft_generation_contract(row)


def test_qwen_adapter_requires_generation_contract_and_preserves_base_model(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema": QWEN_ADAPTER_SCHEMA,
        "track": "graph_reasoner_assisted_pred",
        "dataset": "trace",
        "base_model": "/models/Qwen3-VL-8B-Instruct",
        "output_schema_fields": list(_prediction()),
        "provenance_contract": "predicted_event+rgr_checkpoint",
        "generation_contract": QWEN_GENERATION_CONTRACT,
        "training_contract": {
            "epochs": 1,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "save_steps": 1,
        },
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_qwen_adapter_manifest(tmp_path, expected_track="graph_reasoner_assisted_pred")
    assert loaded["base_model"] == manifest["base_model"]

    other_model = tmp_path / "other_model"
    other_model.mkdir()
    with pytest.raises(ValueError, match="base model"):
        load_qwen_adapter_manifest(tmp_path, expected_base_model=other_model)
    with pytest.raises(ValueError, match="base model"):
        QwenVideoBackend(other_model, adapter=tmp_path)

    del manifest["generation_contract"]
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="generation contract"):
        load_qwen_adapter_manifest(tmp_path)


def test_qwen_adapter_requires_source_timing_before_generation(tmp_path) -> None:
    backend = QwenVideoBackend.__new__(QwenVideoBackend)
    backend.adapter_manifest = QWEN_GENERATION_CONTRACT

    with pytest.raises(ValueError, match="required"):
        backend.generate([tmp_path / "frame.jpg"], "What changed?", [{"shot_id": 1, "frame": 0}])


def test_qwen_training_cli_requires_data_report() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen_lora.py",
            "--model",
            "model",
            "--train-data",
            "train.jsonl",
            "--val-data",
            "val.jsonl",
            "--output-dir",
            "out",
            "--report",
            "report.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--data-report" in result.stderr


def test_qwen_data_report_must_cover_every_sft_row(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.touch()
    val.touch()
    report = {
        "schema": "rallymotionreasoner.qwen_data_report.v1",
        "status": "PASS",
        "track": "graph_reasoner_assisted_pred",
        "media_mode": "videos",
        "gold_events_in_prompt": False,
        "candidate_fallback_used": False,
        "sampling_contract": QWEN_SAMPLING_CONTRACT,
        "splits": {
            "train": {
                "references": 2,
                "prompts": 2,
                "sft_rows": 2,
                "missing_media": 0,
                "leakage": 0,
                "empty_candidates": 0,
                "sft_path": str(train),
            },
            "val": {
                "references": 1,
                "prompts": 1,
                "sft_rows": 1,
                "missing_media": 0,
                "leakage": 0,
                "empty_candidates": 0,
                "sft_path": str(val),
            },
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validate_report = runpy.run_path("scripts/train_qwen_lora.py")["_validate_data_report"]

    with pytest.raises(ValueError, match="coverage/path contract mismatch"):
        validate_report(
            report_path,
            expected_track="graph_reasoner_assisted_pred",
            train_data=train,
            val_data=val,
            train_rows=1,
            val_rows=1,
        )
