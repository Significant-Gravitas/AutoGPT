"""Tests for the delegation tracking context: ETA, criteria, phase timeline.

Every value these helpers handle originates in model-authored JSON or in an
untrusted transcript, so the normalisation and parsing paths are the contract
under test — a malformed ``todos`` payload must degrade to "no phases", never
to an exception on the polling hot path.
"""

from __future__ import annotations

import json

from backend.copilot.model import ChatMessage

from .sub_session_context import (
    MAX_ESTIMATED_MINUTES,
    MAX_SUCCESS_CRITERIA,
    completion_criteria_reminder,
    criteria_preamble,
    normalize_estimated_minutes,
    normalize_success_criteria,
    phases_from_messages,
)


def _todo_call(todos: object, name: str = "TodoWrite") -> dict:
    """An assistant tool call in the persisted OpenAI shape, where
    ``arguments`` is a JSON *string* rather than an object."""
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"todos": todos})},
    }


def _assistant(tool_calls: list[dict] | None) -> ChatMessage:
    return ChatMessage(role="assistant", content="", tool_calls=tool_calls)


def _todo(content: str, status: str) -> dict:
    return {"content": content, "activeForm": f"{content}ing", "status": status}


class TestNormalizeEstimatedMinutes:
    def test_accepts_a_positive_int(self):
        assert normalize_estimated_minutes(15) == 15

    def test_accepts_a_numeric_string(self):
        assert normalize_estimated_minutes("15") == 15

    def test_rejects_zero_and_negatives(self):
        assert normalize_estimated_minutes(0) is None
        assert normalize_estimated_minutes(-5) is None

    def test_rejects_unparseable_values(self):
        assert normalize_estimated_minutes("soon") is None
        assert normalize_estimated_minutes(None) is None
        assert normalize_estimated_minutes({"minutes": 5}) is None

    def test_rejects_booleans(self):
        """``True`` is an int in Python; surfacing it as a 1-minute ETA would
        turn a type confusion into a promise made to the user."""
        assert normalize_estimated_minutes(True) is None

    def test_clamps_absurd_estimates(self):
        assert normalize_estimated_minutes(10**9) == MAX_ESTIMATED_MINUTES


class TestNormalizeSuccessCriteria:
    def test_keeps_non_empty_strings_in_order(self):
        assert normalize_success_criteria(["  runs clean  ", "has tests"]) == [
            "runs clean",
            "has tests",
        ]

    def test_absent_and_empty_both_collapse_to_none(self):
        """``None`` rather than ``[]`` so "no definition of done was given"
        stays distinguishable from "an empty one was"."""
        assert normalize_success_criteria(None) is None
        assert normalize_success_criteria([]) is None
        assert normalize_success_criteria(["", "   "]) is None

    def test_drops_non_string_entries(self):
        assert normalize_success_criteria(["ok", 5, None, {"a": 1}]) == ["ok"]

    def test_rejects_a_bare_string(self):
        assert normalize_success_criteria("runs clean") is None

    def test_caps_the_list(self):
        criteria = normalize_success_criteria([f"c{i}" for i in range(50)])
        assert criteria is not None
        assert len(criteria) == MAX_SUCCESS_CRITERIA


class TestCriteriaPrompting:
    def test_preamble_lists_every_criterion(self):
        block = criteria_preamble(["runs clean", "has tests"])
        assert "Done means ALL of the following are true" in block
        assert "- runs clean" in block
        assert "- has tests" in block

    def test_preamble_is_empty_without_criteria(self):
        """Callers concatenate it unconditionally, so "none" must contribute
        nothing rather than an empty bracketed block."""
        assert criteria_preamble(None) == ""
        assert criteria_preamble([]) == ""

    def test_reminder_restates_the_criteria_at_the_point_of_judgement(self):
        reminder = completion_criteria_reminder(["runs clean", "has tests"])
        assert "2 success criteria" in reminder
        assert "runs clean; has tests" in reminder
        assert "name any that is not met" in reminder

    def test_reminder_is_empty_without_criteria(self):
        assert completion_criteria_reminder(None) == ""


class TestPhasesFromMessages:
    def test_mines_the_latest_todo_write(self):
        phases = phases_from_messages(
            [
                _assistant([_todo_call([_todo("Plan", "in_progress")])]),
                _assistant([_todo_call([_todo("Plan", "completed")])]),
            ]
        )
        assert [(p.content, p.status) for p in phases] == [("Plan", "completed")]

    def test_a_newer_list_wins_over_an_older_one(self):
        """Every TodoWrite call carries the whole plan, so the newest call is
        the current state — an older one must not leak back in."""
        phases = phases_from_messages(
            [
                _assistant([_todo_call([_todo("Old", "pending")])]),
                _assistant(
                    [
                        _todo_call(
                            [_todo("New A", "completed"), _todo("New B", "pending")]
                        )
                    ]
                ),
            ]
        )
        assert [p.content for p in phases] == ["New A", "New B"]

    def test_reads_a_dict_arguments_payload(self):
        """The live-drain shape carries arguments as a dict, not a string."""
        call = {"name": "TodoWrite", "input": {"todos": [_todo("Ship", "pending")]}}
        assert [p.content for p in phases_from_messages([_assistant([call])])] == [
            "Ship"
        ]

    def test_falls_back_to_an_orphaned_result_row(self):
        """The assistant row carrying the arguments is not always persisted —
        the turn's final all-completed update is the one most often lost — so
        the baseline tool's result row is used instead."""
        orphan = ChatMessage(
            role="tool",
            tool_call_id="call-1",
            content=json.dumps(
                {"type": "todo_write", "todos": [_todo("Ship", "completed")]}
            ),
        )
        stale = _assistant([_todo_call([_todo("Ship", "pending")])])
        phases = phases_from_messages([stale, orphan])
        assert [(p.content, p.status) for p in phases] == [("Ship", "completed")]

    def test_ignores_other_tools(self):
        other = _assistant([_todo_call([_todo("X", "pending")], name="run_agent")])
        assert phases_from_messages([other]) == []

    def test_unknown_status_degrades_to_pending(self):
        phases = phases_from_messages(
            [_assistant([_todo_call([{"content": "Ship", "status": "wat"}])])]
        )
        assert phases[0].status == "pending"

    def test_malformed_payloads_yield_no_phases(self):
        malformed = [
            _assistant([{"function": {"name": "TodoWrite", "arguments": "{not json"}}]),
            _assistant([{"function": {"name": "TodoWrite", "arguments": "[]"}}]),
            _assistant([_todo_call({"not": "a list"})]),
            _assistant([_todo_call([{"activeForm": "no content"}, "junk"])]),
            _assistant(None),
        ]
        for message in malformed:
            assert phases_from_messages([message]) == []

    def test_empty_transcript_yields_no_phases(self):
        assert phases_from_messages([]) == []
