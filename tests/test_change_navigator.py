"""
Unit tests for the Change Navigator module.

Run with:
    pytest tests/test_change_navigator.py -v
"""

import json
from datetime import date

import pytest

from autogpt.change_navigator.journal import JournalEntry, JournalStatus, KeyResult
from autogpt.change_navigator.workflow import CheckInWorkflow, _detect_hebrew


# ---------------------------------------------------------------------------
# Journal model tests
# ---------------------------------------------------------------------------


class TestKeyResult:
    def test_valid_creation(self):
        kr = KeyResult(name="Improve NPS", target="60", progress_pct=40)
        assert kr.progress_pct == 40

    def test_invalid_pct_raises(self):
        with pytest.raises(ValueError):
            KeyResult(name="Bad KR", target="x", progress_pct=150)

    def test_to_dict(self):
        kr = KeyResult(name="KR1", target="100%", progress_pct=75, notes="on track")
        d = kr.to_dict()
        assert d["progress_pct"] == 75
        assert d["notes"] == "on track"


class TestJournalEntry:
    def _sample_entry(self) -> JournalEntry:
        return JournalEntry(
            coachee_name="Jane Doe",
            week_number=3,
            entry_date=date(2026, 1, 15),
            central_goal="Launch Q1 digital programme",
            key_results=[
                KeyResult("KR1 – Adoption", "80%", 55),
                KeyResult("KR2 – Revenue", "$1M ARR", 30),
            ],
            obstacles=["Stakeholder resistance", "Budget freeze"],
            inspiration_reflection="Applied 'listen first' with the ops team",
            coach_notes="Stakeholder risk needs immediate attention",
        )

    def test_to_dict_roundtrip(self):
        entry = self._sample_entry()
        d = entry.to_dict()
        restored = JournalEntry.from_dict(d)
        assert restored.central_goal == entry.central_goal
        assert len(restored.key_results) == 2
        assert restored.key_results[0].progress_pct == 55

    def test_to_json_is_valid_json(self):
        entry = self._sample_entry()
        parsed = json.loads(entry.to_json())
        assert parsed["coachee_name"] == "Jane Doe"

    def test_summary_text_contains_key_fields(self):
        entry = self._sample_entry()
        summary = entry.summary_text()
        assert "Jane Doe" in summary
        assert "KR1" in summary
        assert "Stakeholder resistance" in summary
        assert "Q1 digital programme" in summary

    def test_default_status_is_draft(self):
        entry = JournalEntry(coachee_name="Test", week_number=1)
        assert entry.status == JournalStatus.DRAFT


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectHebrew:
    def test_hebrew_text(self):
        assert _detect_hebrew("מה שלומך?") is True

    def test_english_text(self):
        assert _detect_hebrew("Hello world") is False

    def test_mixed_text(self):
        assert _detect_hebrew("My name is דוד") is True


# ---------------------------------------------------------------------------
# Workflow tests (no LLM)
# ---------------------------------------------------------------------------


class TestCheckInWorkflow:
    def _make_workflow(self) -> CheckInWorkflow:
        return CheckInWorkflow(coachee_name="Test User", week_number=7)

    def test_start_returns_string(self):
        wf = self._make_workflow()
        q = wf.start()
        assert isinstance(q, str)
        assert len(q) > 10

    def test_full_english_flow(self):
        wf = self._make_workflow()
        wf.start()

        # Stage: opening
        reply = wf.advance("Launch the new CRM platform")
        assert isinstance(reply, str)

        # Stage: key results (one KR + done)
        reply = wf.advance("Migrate 50% of accounts — 60%")
        assert "60%" in reply or "60" in reply
        reply = wf.advance("done")

        # Stage: obstacles
        reply = wf.advance("Integration issues with legacy ERP")

        # Stage: reflection
        reply = wf.advance("Applied active listening in Monday's all-hands")

        # Stage: approval — agent shows summary
        assert wf._stage.name == "APPROVAL"
        assert not wf.is_done()

        # Approve
        reply = wf.advance("send")
        assert wf.is_done()

        entry = wf.journal_entry
        assert entry is not None
        assert entry.status == JournalStatus.APPROVED
        assert entry.coachee_name == "Test User"
        assert entry.week_number == 7

    def test_full_hebrew_flow(self):
        wf = self._make_workflow()
        wf.start()

        # Opening in Hebrew triggers Hebrew responses
        reply = wf.advance("להשיק את מערכת ה-CRM החדשה")
        # Agent should respond in Hebrew
        assert _detect_hebrew(reply)

        # KR + done in Hebrew
        wf.advance("שיפור שביעות רצון לקוח — 45%")
        wf.advance("סיימתי")

        # Obstacles
        wf.advance("התנגדות מצד בכירים")

        # Reflection
        wf.advance("יישמתי 'קשיבות' בפגישת הצוות")

        # Approve in Hebrew
        reply = wf.advance("שלח")
        assert wf.is_done()
        assert wf.journal_entry.status == JournalStatus.APPROVED

    def test_approval_edit_then_send(self):
        """User requests a change at approval stage, then approves."""
        wf = self._make_workflow()
        wf.start()
        wf.advance("Launch mobile app")
        wf.advance("App downloads — 80%")
        wf.advance("done")
        wf.advance("No obstacles this week")
        wf.advance("Stayed focused on outcomes over outputs")

        # Request edit
        wf.advance("Actually the KR progress is 90%, not 80%")
        assert not wf.is_done()

        # Now approve
        wf.advance("send")
        assert wf.is_done()

    def test_kr_parsing(self):
        from autogpt.change_navigator.workflow import CheckInWorkflow

        kr = CheckInWorkflow._parse_kr("Increase revenue by Q2 — 35%", index=0)
        assert kr["progress_pct"] == 35

        kr_no_pct = CheckInWorkflow._parse_kr("Ship v2.0 feature", index=2)
        assert kr_no_pct["progress_pct"] == 0
        assert kr_no_pct["name"] != ""
