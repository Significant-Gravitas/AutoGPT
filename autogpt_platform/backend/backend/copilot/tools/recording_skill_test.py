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

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from .recording_models import WorkflowRecording
from .recording_skill import (
    dry_run,
    generate_skill_from_recording,
    propose_clarifications,
)


def _step(
    seq,
    action,
    label=None,
    raw=None,
    vtype="text",
    selectors=None,
    cursor=None,
    role=None,
):
    enrichment = {"kind": "dom" if selectors else "none"}
    if label:
        enrichment["label"] = label
    if role:
        enrichment["role"] = role
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

    def test_a_constant_confirmed_by_field_key_drops_the_parameter(self):
        """`confirmations.get(key) or confirmations.get(label)` turned a False
        given by key into None, so the field stayed an unconfirmed candidate."""
        skill = generate_skill_from_recording(
            _single_row_recording(),
            clarifications={
                "confirmed_parameters": {"fill|First Name": True, "fill|Email": False}
            },
        )

        assert skill.needs_confirmation is False
        assert {p.label for p in skill.parameters} == {"First Name"}

    @pytest.mark.parametrize(
        "clarifications",
        [
            {"confirmed_parameters": None, "data_source_columns": None},
            {"confirmed_parameters": ["Email"], "data_source_columns": "Email"},
            {
                "confirmed_parameters": {"Email": "no"},
                "data_source_columns": {"Email": 3},
            },
        ],
        ids=["null", "not-a-map", "mistyped-answers"],
    )
    def test_an_unreadable_confirmation_map_is_no_answer(self, clarifications):
        """`.get(..., {})` returned a null or a list as it was, and the next
        `.get` on it raised AttributeError."""
        skill = generate_skill_from_recording(
            _single_row_recording(), clarifications=clarifications
        )

        assert skill.needs_confirmation is True
        assert not any(p.confirmed for p in skill.parameters)


def _joined_recording():
    return WorkflowRecording.from_payload(
        {
            "recording_id": "rec_join",
            "steps": [
                _step(
                    1,
                    "fill",
                    label="Email",
                    raw="john@x.com",
                    vtype="email",
                    selectors=[{"strategy": "id", "value": "#email"}],
                ),
                _step(
                    2,
                    "fill",
                    label="Confirm email",
                    raw="john@x.com",
                    vtype="email",
                    selectors=[{"strategy": "id", "value": "#confirm"}],
                ),
                _step(
                    3,
                    "fill",
                    label="Contact",
                    raw="Jane",
                    selectors=[{"strategy": "id", "value": "#contact"}],
                ),
                _step(4, "submit", label="Save"),
            ],
        }
    )


class TestJoinedColumnNames:
    def test_no_field_takes_a_joined_columns_name(self):
        """A joined column's parameter is named after the column, which is how
        a data row fills it, but the name wasn't reserved: a field with the
        same name got the same parameter name, and the two collided."""
        skill = generate_skill_from_recording(
            _joined_recording(),
            clarifications={
                "data_source_columns": {"Contact": "email"},
                "confirmed_parameters": {"Email": True, "Confirm email": False},
            },
        )

        names = [p.name for p in skill.parameters]
        assert len(names) == len(set(names))
        by_label = {p.label: p.name for p in skill.parameters}
        assert by_label == {"Email": "email_2", "Contact": "email"}

    def test_two_fields_joined_to_one_column_share_its_parameter(self):
        skill = generate_skill_from_recording(
            _joined_recording(),
            clarifications={
                "data_source_columns": {"Email": "email", "Confirm email": "email"},
                "confirmed_parameters": {"Contact": False},
            },
        )

        assert [p.name for p in skill.parameters] == ["email"]
        bound = {b.seq: b.parameter for b in skill.manifest.bindings if b.parameter}
        assert bound == {1: "email", 2: "email"}


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

    def test_what_otto_gets_carries_no_parameter_value(self):
        """Stripping cleared the recording, but each parameter kept the
        demonstrated values as `sample_values`, which the tool result
        returned and the session kept."""
        skill = generate_skill_from_recording(_two_row_recording())

        shown = json.dumps(skill.to_dict())

        assert {p.name for p in skill.parameters} >= {"first_name", "email"}
        for value in ("John", "Jane", "john@x.com", "jane@y.com"):
            assert value not in shown


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


def _mock_shim(*, is_error: bool = False):
    shim = MagicMock()
    shim.computer = MagicMock()
    shim.computer.call = AsyncMock(
        return_value={
            "is_error": is_error,
            "content": [{"type": "text", "text": "refused" if is_error else "ok"}],
        }
    )
    return shim


def _calls(shim, tool: str) -> list:
    return [c for c in shim.computer.call.await_args_list if c.args[0] == tool]


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
        # Each data row fills the demonstrated row's fields once, in order.
        typed = [c.args[1]["text"] for c in _calls(shim, "type_text")]
        assert typed == ["Alice", "alice@x.com", "Bob", "bob@y.com"]

    @pytest.mark.asyncio
    async def test_a_repeated_demonstration_replays_one_row_per_data_row(self):
        """The manifest held both demonstrated rows, so each data row was
        typed twice and, with actions allowed, submitted twice."""
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim()
        rows = [
            {"first_name": "Alice", "email": "alice@x.com"},
            {"first_name": "Bob", "email": "bob@y.com"},
        ]

        result = await dry_run(skill, rows, shim=shim, allow_destructive=True)

        typed = [c.args[1]["text"] for c in _calls(shim, "type_text")]
        assert typed == ["Alice", "alice@x.com", "Bob", "bob@y.com"]
        assert len(_calls(shim, "click")) == 2  # one submit per data row
        assert result.rows_ok == 2
        assert [b.action for b in skill.manifest.bindings] == [
            "fill",
            "fill",
            "submit",
        ]
        assert skill.skill_md.count("set First Name") == 1

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
        # A held-back last step leaves nothing untested.
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_a_dry_run_clicks_only_into_fields_until_actions_are_allowed(
        self,
    ):
        """A click isn't a destructive verb, so a dry run without
        allow_destructive clicked a recorded button at its screen point,
        though it could be Delete. A click into a text field only focuses it,
        and the fill after it needs that focus."""
        recording = WorkflowRecording.from_payload(
            {
                "recording_id": "rec_click",
                "steps": [
                    _step(1, "click", label="First Name", role="textbox"),
                    _step(2, "fill", label="First Name", raw="John"),
                    _step(3, "click", label="Delete", role="button"),
                    _step(4, "fill", label="Note", raw="x"),
                    _step(5, "submit", label="Save"),
                ],
            }
        )
        skill = generate_skill_from_recording(
            recording,
            clarifications={
                "confirmed_parameters": {"First Name": True, "Note": False}
            },
        )
        held_back, allowed = _mock_shim(), _mock_shim()

        result = await dry_run(skill, [{"first_name": "Ada"}], shim=held_back)
        await dry_run(
            skill, [{"first_name": "Ada"}], shim=allowed, allow_destructive=True
        )

        assert [c.args[1]["x"] for c in _calls(held_back, "click")] == [101]
        assert [c.args[1]["text"] for c in _calls(held_back, "type_text")] == ["Ada"]
        assert result.destructive_blocked is True
        # The steps after it would act on a screen the recording never saw.
        (row,) = result.per_row
        assert (row.steps[-1].action, row.steps[-1].detail) == (
            "click",
            "destructive_skipped",
        )
        # So the row proves nothing about them, and the dry run isn't a pass:
        # it reported ok though only the first fill had been tested.
        assert row.ok is False
        assert row.error is not None and "2 step(s) after it" in row.error
        assert result.ok is False
        assert [c.args[1]["x"] for c in _calls(allowed, "click")] == [101, 103, 105]

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
        # click was used for the submit, at the recorded screen point.
        clicks = _calls(shim, "click")
        assert clicks
        assert clicks[0].args[1]["target"] == {
            "kind": "desktop",
            "display_id": "primary",
        }

    @pytest.mark.asyncio
    async def test_a_step_the_dry_run_cannot_replay_fails_its_row(self):
        """A navigate step is not replayed at the floor level. Counting it as
        passed would report a row as verified when part of it never ran."""
        recording = WorkflowRecording.from_payload(
            {
                "recording_id": "rec_nav",
                "steps": [
                    _step(1, "navigate", label="Open the form"),
                    _step(
                        2,
                        "fill",
                        label="First Name",
                        raw="John",
                        selectors=[{"strategy": "id", "value": "#first_name"}],
                    ),
                    _step(
                        3,
                        "navigate",
                        label="Open the form",
                    ),
                    _step(
                        4,
                        "fill",
                        label="First Name",
                        raw="Jane",
                        selectors=[{"strategy": "id", "value": "#first_name"}],
                    ),
                ],
            }
        )
        skill = generate_skill_from_recording(recording)
        result = await dry_run(skill, [{"first_name": "Alice"}], shim=_mock_shim())

        navigate = [
            s for r in result.per_row for s in r.steps if s.action == "navigate"
        ]
        assert navigate and all(not s.ok for s in navigate)
        assert all("not executed" in (s.detail or "") for s in navigate)
        assert result.rows_ok == 0
        assert result.ok is False

    @pytest.mark.asyncio
    async def test_missing_parameter_in_row_fails_that_row(self):
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim()
        result = await dry_run(skill, [{"first_name": "A"}], shim=shim)
        assert result.rows_ok == 0
        assert "missing parameter" in (result.per_row[0].error or "")

    @pytest.mark.asyncio
    async def test_a_fill_is_reported_unasserted_until_there_is_a_readback(self):
        skill = generate_skill_from_recording(_two_row_recording())
        shim = _mock_shim()
        result = await dry_run(
            skill, [{"first_name": "Alice", "email": "alice@x.com"}], shim=shim
        )
        fn_steps = [s for r in result.per_row for s in r.steps if s.seq == 1]
        assert fn_steps and fn_steps[0].asserted is False and fn_steps[0].ok is True

    @pytest.mark.asyncio
    async def test_input_the_driver_refuses_fails_the_step(self):
        skill = generate_skill_from_recording(_two_row_recording())
        result = await dry_run(
            skill,
            [{"first_name": "Alice", "email": "alice@x.com"}],
            shim=_mock_shim(is_error=True),
        )
        fn_steps = [s for r in result.per_row for s in r.steps if s.seq == 1]
        assert fn_steps and fn_steps[0].ok is False
        assert "refused" in (fn_steps[0].detail or "")
