from types import SimpleNamespace

from rallymotionreasoner.event_detection.decoder import DecodedEvent
from rallymotionreasoner.pipeline import RallyMotionReasoner, select_frame_indices


def test_frame_selection_preserves_evidence_centers_within_video_budget() -> None:
    candidates = [{"frame": frame} for frame in range(10, 90, 10)]
    selected = select_frame_indices(100, candidates, max_global_frames=16, local_frames_per_candidate=3)

    assert len(selected) <= 32
    assert selected == sorted(set(selected))
    assert set(range(10, 90, 10)).issubset(selected)
    assert 0 in selected and 99 in selected


def test_low_evidence_abstains_without_calling_qwen(tmp_path) -> None:
    for frame in range(3):
        (tmp_path / f"frame_{frame:06d}.jpg").touch()
    runtime = SimpleNamespace(
        events=[DecodedEvent(1, 0.04, 0.9, {}, {})],
        frame_features=None,
        frame_indices=[0, 1, 2],
        feature_provenance=SimpleNamespace(backend="test", tracknet_source="test"),
    )

    class Event:
        backend = "motion_region"

        def predict(self, *_args, **_kwargs):
            return runtime

    class Selector:
        name = "test"
        last_predictions = {}

        def select(self, *_args):
            return [{"shot_id": 1, "frame": 1, "selected": False, "confidence": 0.1}]

    class Qwen:
        adapter_manifest = None

        def generate(self, *_args, **_kwargs):
            raise AssertionError("Qwen must not answer without selected evidence")

    model = RallyMotionReasoner.__new__(RallyMotionReasoner)
    model.event, model.selector, model.qwen, model.paths = Event(), Selector(), Qwen(), {}
    answer = model.predict(tmp_path, "Which shot mattered?", fps=25.0, ball_track={})

    assert answer["answerability"] == "unanswerable"
    assert answer["answer"] == ""
    assert answer["evidence_shot_ids"] == []
    assert answer["provenance"]["abstention_reason"] == "low_evidence"
