import json

from tennisvar.generation.qwen import parse_qwen_json, qwen_prompt


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
