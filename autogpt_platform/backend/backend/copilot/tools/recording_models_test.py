"""Tests for the workflow-recording wire-schema dataclasses."""

from __future__ import annotations

from .recording_models import (
    DESTRUCTIVE_VERBS,
    MUTATING_VERBS,
    RecordingSummary,
    StepEnrichment,
    StepValue,
    TrajectoryStep,
    WorkflowRecording,
)


class TestStepEnrichment:
    def test_defaults_to_none_kind(self):
        e = StepEnrichment.from_payload(None)
        assert e.kind == "none"
        assert e.selectors == []

    def test_parses_dom_selectors_most_stable_first(self):
        e = StepEnrichment.from_payload(
            {
                "kind": "dom",
                "selectors": [
                    {"strategy": "id", "value": "#fn"},
                    {"strategy": "label", "value": "First Name"},
                ],
                "label": "First Name",
                "url": "https://x/new",
            }
        )
        assert e.kind == "dom"
        assert [s["strategy"] for s in e.selectors] == ["id", "label"]
        assert e.url == "https://x/new"

    def test_drops_malformed_selectors(self):
        e = StepEnrichment.from_payload(
            {"selectors": [{"strategy": "id"}, "garbage", {"value": "x"}]}
        )
        assert e.selectors == []


class TestStepValue:
    def test_none_payload_returns_none(self):
        assert StepValue.from_payload(None) is None

    def test_is_parameter_unconfirmed_is_none(self):
        v = StepValue.from_payload({"raw": "John", "type": "text"})
        assert v is not None
        assert v.is_parameter is None  # §8 — must be confirmed, not guessed


class TestTrajectoryStep:
    def test_mutating_and_destructive_classification(self):
        fill = TrajectoryStep.from_payload({"action": "fill"})
        submit = TrajectoryStep.from_payload({"action": "submit"})
        click = TrajectoryStep.from_payload({"action": "click"})
        assert fill.is_mutating and not fill.is_destructive
        assert submit.is_mutating and submit.is_destructive
        assert not click.is_mutating and not click.is_destructive

    def test_malformed_cursor_becomes_none(self):
        s = TrajectoryStep.from_payload({"action": "click", "cursor": [1]})
        assert s.cursor is None

    def test_roundtrip_to_dict(self):
        s = TrajectoryStep.from_payload(
            {
                "seq": 3,
                "action": "fill",
                "cursor": [10, 20],
                "value": {"raw": "x", "type": "text", "is_parameter": True},
            }
        )
        d = s.to_dict()
        assert d["seq"] == 3
        assert d["cursor"] == [10, 20]
        assert d["value"]["is_parameter"] is True


class TestWorkflowRecording:
    def test_sorts_steps_by_seq_when_all_present(self):
        rec = WorkflowRecording.from_payload(
            {
                "recording_id": "rec_1",
                "steps": [
                    {"seq": 2, "action": "submit"},
                    {"seq": 1, "action": "fill"},
                ],
            }
        )
        assert [s.seq for s in rec.steps] == [1, 2]

    def test_default_route_is_extract_then_cloud(self):
        rec = WorkflowRecording.from_payload({"recording_id": "rec_1"})
        assert rec.interpretation_route == "extract_then_cloud"


class TestRecordingSummary:
    def test_parses_coverage_and_skips_bad_values(self):
        s = RecordingSummary.from_payload(
            {
                "recording_id": "rec_1",
                "step_count": 14,
                "enrichment_coverage": {"dom": 11, "ax": "bad", "none": 3},
                "duration_seconds": 47.2,
            }
        )
        assert s.enrichment_coverage == {"dom": 11, "none": 3}


def test_verb_sets_are_consistent():
    # Every destructive verb is also a mutating verb.
    assert DESTRUCTIVE_VERBS.issubset(MUTATING_VERBS)
