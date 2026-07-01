"""Tests for recording → SKILL.md generalization + the trust loop.

The load-bearing spec rules under test:
  * §8 — parameter inference is confirmed, never guessed. A single-row
    recording with unconfirmed params MUST NOT auto-save.
  * ≥2 rows → varied value is a parameter (confirmed); constant value is a
    constant (confirmed).
  * §7 — value-stripping drops parameter raw values; selector fall-through
    always ends in visual.
  * §9 — multi-row dry-run with read-back asserts; destructive submits
    gated behind an explicit flag.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .recording_models import WorkflowRecording
from .recording_skill import (
    dry_run,
    generate_skill_from_recording,
    propose_clarifications,
)


def _step(seq, action, label=None, raw=None, vtype="text", selectors=None, cursor=None):
    enrichment = {"kind": "dom" if selectors else "none"}
    if label:
        enrichment["label"] = label
    if selectors:
        enrichment["selectors"] = selectors
    payload = {
        "seq": seq,
        "action": action,
        "screenshot_ref": f"stub_{seq}",
        "cursor": cursor or [100 + seq, 200 + seq],
        "active_app": "Google Chrome",
        "enrichment": enrichment,
    }
    if raw is not None or action in ("fill", "select"):
        payload["value"] = {"raw": raw, "type": vtype, "is_parameter": None}
    return payload


def _single_row_recording():
    return WorkflowRecording.from_payload(
        {
            "recording_id": "rec_1",
            "steps": [
                _step(
                    1,
                    "fill",
                    label="First Name",
                    raw="John",
                    selectors=[{"strategy": "id", "value": "#first_name"}],
                ),
                _step(
                    2,
                    "fill",
                    label="Email",
                    raw="john@x.com",
                    vtype="email",
                    selectors=[{"strategy": "id", "value": "#email"}],
                ),
                _step(3, "submit", label="Save", selectors=None),
            ],
        }
    )


def _two_row_recording(*, email_constant=False):
    email1 = "shared@x.com" if email_constant else "john@x.com"
    email2 = "shared@x.com" if email_constant else "jane@y.com"
    return WorkflowRecording.from_payload(
        {
            "recording_id": "rec_2",
            "steps": [
                _step(
                    1,
                    "fill",
                    label="First Name",
                    raw="John",
                    selectors=[{"strategy": "id", "value": "#first_name"}],
                ),
                _step(
                    2,
                    "fill",
                    label="Email",
                    raw=email1,
                    vtype="email",
                    selectors=[{"strategy": "id", "value": "#email"}],
                ),
                _step(3, "submit", label="Save"),
                _step(
                    4,
                    "fill",
                    label="First Name",
                    raw="Jane",
                    selectors=[{"strategy": "id", "value": "#first_name"}],
                ),
                _step(
                    5,
                    "fill",
                    label="Email",
                    raw=email2,
                    vtype="email",
                    selectors=[{"strategy": "id", "value": "#email"}],
                ),
                _step(6, "submit", label="Save"),
            ],
        }
    )


class TestSingleRowGate:
    """§8 — a single-row recording must NOT auto-produce a final skill."""

    def test_single_row_blocks_with_questions(self):
        skill = generate_skill_from_recording(_single_row_recording())
        assert skill.needs_confirmation is True
        assert skill.questions, "must surface clarifying questions"
        assert skill.needs_second_row is True
        # Draft md exists but the gate says don't save.
        assert skill.skill_md

    def test_single_row_questions_name_the_fields(self):
        skill = generate_skill_from_recording(_single_row_recording())
        joined = " ".join(q.question for q in skill.questions)
        assert "First Name" in joined
        assert "Email" in joined

    def test_field_semantics_is_prior_not_autosave(self):
        """An Email field is a candidate by prior — but still unconfirmed,
        so the skill does not save on the strength of the prior alone."""
        skill = generate_skill_from_recording(_single_row_recording())
        email = [p for p in skill.parameters if p.label == "Email"]
        assert email and email[0].confirmed is False
        assert skill.needs_confirmation is True


class TestCoPilotConfirmation:
    def test_explicit_confirmation_unblocks(self):
        skill = generate_skill_from_recording(
            _single_row_recording(),
            clarifications={
                "confirmed_parameters": {"First Name": True, "Email": True}
            },
        )
        assert skill.needs_confirmation is False
        assert all(p.confirmed for p in skill.parameters)

    def test_explicit_constant_drops_parameter(self):
        skill = generate_skill_from_recording(
            _single_row_recording(),
            clarifications={
                "confirmed_parameters": {"First Name": True, "Email": False}
            },
        )
        assert skill.needs_confirmation is False
        names = {p.label for p in skill.parameters}
        assert "First Name" in names
        assert "Email" not in names


class TestMultiRowInference:
    def test_two_rows_varied_value_is_confirmed_parameter(self):
        skill = generate_skill_from_recording(_two_row_recording())
        assert skill.needs_confirmation is False
        labels = {p.label for p in skill.parameters}
        assert "First Name" in labels and "Email" in labels
        assert all(p.confirmed for p in skill.parameters)

    def test_two_rows_constant_value_becomes_constant(self):
        """A value that stayed the same across rows is a constant, not a
        parameter — this is exactly what a single-row dry-run would miss."""
        skill = generate_skill_from_recording(_two_row_recording(email_constant=True))
        assert skill.needs_confirmation is False
        labels = {p.label for p in skill.parameters}
        assert "First Name" in labels
        assert "Email" not in labels  # constant across rows


class TestValueStripping:
    def test_parameter_raw_values_are_stripped(self):
        rec = _single_row_recording()
        generate_skill_from_recording(
            rec, clarifications={"confirmed_parameters": {"First Name": True}}
        )
        first_name_step = next(s for s in rec.steps if s.seq == 1)
        assert first_name_step.value is not None
        assert first_name_step.value.raw is None  # §7 stripped

    def test_skill_md_does_not_leak_parameter_value(self):
        skill = generate_skill_from_recording(
            _single_row_recording(),
            clarifications={
                "confirmed_parameters": {"First Name": True, "Email": True}
            },
        )
        assert "John" not in skill.skill_md
        assert "john@x.com" not in skill.skill_md


class TestManifestFallthrough:
    def test_selector_fallthrough_ends_in_visual(self):
        skill = generate_skill_from_recording(
            _single_row_recording(),
            clarifications={"confirmed_parameters": {"First Name": True}},
        )
        for binding in skill.manifest.bindings:
            chain = binding.selector_fallthrough
            assert chain[-1]["via"] == "visual"
        # DOM step has dom first, visual last.
        fill_binding = next(b for b in skill.manifest.bindings if b.seq == 1)
        assert fill_binding.selector_fallthrough[0]["via"] == "dom"

    def test_destructive_flag_set_when_submit_present(self):
        skill = generate_skill_from_recording(
            _single_row_recording(),
            clarifications={"confirmed_parameters": {"First Name": True}},
        )
        assert skill.destructive is True
        submit_binding = next(
            b for b in skill.manifest.bindings if b.action == "submit"
        )
        assert submit_binding.destructive is True


class TestProposeClarifications:
    def test_includes_data_source_and_error_policy(self):
        qs = propose_clarifications(_single_row_recording())
        kinds = {q.kind for q in qs}
        assert "data_source" in kinds
        assert "error_policy" in kinds  # destructive recording
        assert "parameter" in kinds


def _mock_shim(clipboard_value=None):
    shim = MagicMock()
    shim.computer = MagicMock()
    shim.computer.type = AsyncMock()
    shim.computer.click = AsyncMock()
    if clipboard_value is not None:
        shim.computer.clipboard_read = AsyncMock(return_value=clipboard_value)
    else:
        shim.computer.clipboard_read = AsyncMock(return_value=None)
    return shim


class TestDryRun:
    @pytest.mark.asyncio
    async def test_multi_row_dry_run_drives_each_row(self):
        skill = generate_skill_from_recording(
            _two_row_recording(),
        )
        shim = _mock_shim()
        rows = [
            {"first_name": "Alice", "email": "alice@x.com"},
            {"first_name": "Bob", "email": "bob@y.com"},
        ]
        result = await dry_run(skill, rows, shim=shim)
        assert result.rows_attempted == 2
        # type() was called for the parameter fills of both rows.
        assert shim.computer.type.await_count >= 2

    @pytest.mark.asyncio
    async def test_destructive_submit_gated_by_default(self):
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim()
        result = await dry_run(
            skill, [{"first_name": "A", "email": "a@x.com"}], shim=shim
        )
        assert result.destructive_blocked is True
        # submit steps are recorded as skipped, not clicked.
        submit_steps = [
            s for r in result.per_row for s in r.steps if s.action == "submit"
        ]
        assert submit_steps and all(
            s.detail == "destructive_skipped" for s in submit_steps
        )

    @pytest.mark.asyncio
    async def test_destructive_runs_when_allowed(self):
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim()
        result = await dry_run(
            skill,
            [{"first_name": "A", "email": "a@x.com"}],
            shim=shim,
            allow_destructive=True,
        )
        assert result.destructive_blocked is False
        # click() was used for the submit.
        assert shim.computer.click.await_count >= 1

    @pytest.mark.asyncio
    async def test_missing_parameter_in_row_fails_that_row(self):
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim()
        result = await dry_run(skill, [{"first_name": "A"}], shim=shim)
        assert result.rows_ok == 0
        assert "missing parameter" in (result.per_row[0].error or "")

    @pytest.mark.asyncio
    async def test_readback_assert_passes_when_value_matches(self):
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim(clipboard_value="Alice")
        result = await dry_run(
            skill, [{"first_name": "Alice", "email": "alice@x.com"}], shim=shim
        )
        # First-name fill asserted; clipboard returned "Alice" so it matches.
        fn_steps = [s for r in result.per_row for s in r.steps if s.seq == 1]
        assert fn_steps and fn_steps[0].asserted is True and fn_steps[0].ok is True
