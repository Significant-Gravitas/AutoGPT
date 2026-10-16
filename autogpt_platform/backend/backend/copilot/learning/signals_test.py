"""Acceptance corpus for deterministic outcome signals.

Only unambiguous user reports and typed tool outcomes may become
verifying signals. The negative corpus pins the wording that must never
count: negations, questions, conditionals, quoted examples, and tool
results that merely returned without an error.
"""

from __future__ import annotations

import json

import pytest

from backend.copilot.model import ChatMessage

from .contract import VERIFYING_SIGNAL_KINDS
from .signals import extract_turn_signals, is_learn_request, user_signals


def _kinds(text: str) -> set[str]:
    return {s.kind for s in user_signals(text, "msg:1")}


@pytest.mark.parametrize(
    "text",
    [
        "That worked, thanks.",
        "Perfect. It worked on the second run.",
        "I checked the output and it works.",
        "The import succeeded with 1,204 rows.",
    ],
)
def test_explicit_confirmations_count(text: str) -> None:
    assert "user_confirmation" in _kinds(text)


@pytest.mark.parametrize(
    "text",
    [
        "That has not worked yet.",
        "It hasn't worked so far — try again?",
        "If that worked we can move on.",
        "Has this been approved?",
        "Did it work?",
        'The doc says "that worked" but I never ran it.',
        "Once it worked before, but not now.",
        "Not approved. Please revise.",
        "Not verified yet.",
        "I don't think that worked.",
        "It never worked.",
        "`that worked` is the phrase we grep for",
    ],
)
def test_negated_questioning_conditional_or_quoted_wording_never_counts(
    text: str,
) -> None:
    assert not _kinds(text) & VERIFYING_SIGNAL_KINDS


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I approve this, go ahead.", True),
        ("Approved, thanks!", True),
        ("Ship it.", True),
        ("Has this been approved by legal?", False),
        ("Not approved.", False),
        ("Approved? Not yet.", False),
        ("Whether approved or not, keep going.", False),
    ],
)
def test_acceptance_requires_explicit_unqualified_wording(
    text: str, expected: bool
) -> None:
    assert ("accepted_artifact" in _kinds(text)) is expected


def test_learn_requests_are_explicit_and_not_questions() -> None:
    assert is_learn_request("Great — save this as a skill.")
    assert is_learn_request("Please learn this so next time is faster")
    assert not is_learn_request("Should I save this as a skill?")
    assert not is_learn_request("Don't save this as a skill.")


def _tool(seq: int, payload: dict) -> ChatMessage:
    return ChatMessage(role="tool", content=json.dumps(payload), sequence=seq)


def test_only_typed_checkable_tool_outcomes_become_signals() -> None:
    messages = [
        ChatMessage(role="user", content="Please import the CSV", sequence=1),
        _tool(2, {"type": "skill_loaded", "name": "csv-import"}),
        _tool(3, {"type": "web_fetch", "content": "<html/>", "status_code": 200}),
        _tool(4, {"type": "agent_builder_clarification_needed", "questions": []}),
        _tool(5, {"type": "execution_started", "status": "QUEUED"}),
        _tool(6, {"type": "block_output", "success": True, "outputs": {"rows": [42]}}),
        _tool(7, {"type": "bash_exec", "exit_code": 1, "stdout": "", "stderr": "boom"}),
        _tool(8, {"type": "error", "message": "failed"}),
        ChatMessage(role="assistant", content="Done: imported 42 rows.", sequence=9),
    ]
    refs, signals = extract_turn_signals(messages)
    assert {r.ref for r in refs} == {f"msg:{i}" for i in range(1, 10)}
    by_ref = {s.ref: s.kind for s in signals}
    assert by_ref == {
        "msg:6": "tool_result",
        "msg:7": "tool_error",
        "msg:8": "tool_error",
    }


def test_question_only_and_plan_only_turns_yield_no_verifying_signal() -> None:
    messages = [
        ChatMessage(role="user", content="How would you import this CSV?", sequence=1),
        _tool(2, {"type": "task_decomposition", "steps": [{"description": "plan"}]}),
        ChatMessage(role="assistant", content="Here is the plan. Done!", sequence=3),
    ]
    _, signals = extract_turn_signals(messages)
    assert not {s.kind for s in signals} & VERIFYING_SIGNAL_KINDS


def test_completed_agent_run_counts_but_queued_run_does_not() -> None:
    completed = _tool(
        2, {"type": "agent_output", "execution": {"status": "COMPLETED", "outputs": {}}}
    )
    failed = _tool(3, {"type": "agent_output", "execution": {"status": "FAILED"}})
    _, signals = extract_turn_signals([completed, failed])
    assert [(s.ref, s.kind) for s in signals] == [
        ("msg:2", "tool_result"),
        ("msg:3", "tool_error"),
    ]
