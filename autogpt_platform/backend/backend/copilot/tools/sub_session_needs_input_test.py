"""A child that stops on ``ask_question`` reports ``needs_input``, not done."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.model import PendingQuestion
from backend.copilot.sdk.session_waiter import SessionResult
from backend.copilot.sdk.stream_accumulator import ToolCallEntry

from .get_sub_session_result import GetSubSessionResultTool
from .run_sub_session import response_from_outcome


def _asked(*questions: dict) -> SessionResult:
    return SessionResult(
        response_text="I need one thing before I start.",
        tool_calls=[
            ToolCallEntry(tool_call_id="t1", tool_name="read_notion", input={}),
            ToolCallEntry(
                tool_call_id="t2",
                tool_name="ask_question",
                input={"questions": list(questions)},
                output="{}",
                success=True,
            ),
        ],
    )


def _respond(result: SessionResult):
    return response_from_outcome(
        outcome="completed",
        result=result,
        inner_session_id="inner-1",
        parent_session_id="parent",
        elapsed=3.0,
        workspace_files=[],
        actor="Alex",
    )


def test_a_turn_ending_on_ask_question_needs_the_user():
    r = _respond(
        _asked(
            {"question": "Which release?", "options": ["Q4", "December", "Both"]},
            {"question": "Who signs off?"},
        )
    )
    assert r.status == "needs_input"
    assert r.question == "Which release?; Who signs off?"
    assert r.question_options == ["Q4", "December", "Both"]
    assert r.response == "I need one thing before I start."
    assert "Alex" in r.message and "user" in r.message


def test_a_replayed_ask_with_json_arguments_is_still_a_question():
    result = _asked()
    result.tool_calls[-1] = ToolCallEntry(
        tool_call_id="t2",
        tool_name="ask_question",
        input='{"questions": [{"question": "Which repo?"}]}',
    )
    r = _respond(result)
    assert r.status == "needs_input"
    assert r.question == "Which repo?"
    assert r.question_options == []


def test_an_ask_earlier_in_the_turn_is_not_a_stop():
    result = _asked({"question": "Which release?"})
    result.tool_calls.append(
        ToolCallEntry(tool_call_id="t3", tool_name="write_doc", input={})
    )
    assert _respond(result).status == "completed"


@pytest.mark.asyncio
async def test_a_cold_poll_reads_the_parked_question(monkeypatch):
    """Polled after the child went idle, the parked question is the signal:
    the persisted last message may be plain text with no tool call."""
    sub = MagicMock(user_id="alice", expert_id=None)
    sub.metadata.delegated_by_session_id = None
    sub.metadata.delegation_cap_usd = None
    sub.metadata.pending_question = PendingQuestion(
        text="Which release?", asked_at=datetime.now(UTC), options=["Q4"]
    )
    last = MagicMock(role="assistant", content="Which release?", tool_calls=None)
    last.created_at = datetime.now(UTC)
    sub.messages = [last]
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.get_chat_session",
        AsyncMock(return_value=sub),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.stream_registry.get_session",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.list_sub_workspace_files",
        AsyncMock(return_value=[]),
    )
    parent = MagicMock(session_id="parent", expert_id=None)
    r = await GetSubSessionResultTool()._execute(
        user_id="alice", session=parent, sub_session_id="inner-1"
    )
    assert r.status == "needs_input"
    assert (r.question, r.question_options) == ("Which release?", ["Q4"])


@pytest.mark.asyncio
async def test_an_idle_thread_parked_on_a_question_needs_no_wait(monkeypatch):
    """A turn can end on the ask_question result row itself, with no closing
    text. The parked question still says the thread is waiting on the user,
    so the poll answers at once instead of waiting on a stream that ended."""
    sub = MagicMock(user_id="alice", expert_id=None)
    sub.metadata.delegated_by_session_id = None
    sub.metadata.delegation_cap_usd = None
    sub.metadata.pending_question = PendingQuestion(
        text="Which repo?", asked_at=datetime.now(UTC)
    )
    tool_row = MagicMock(role="tool", content="{}", tool_calls=None)
    tool_row.created_at = datetime.now(UTC)
    sub.messages = [tool_row]
    wait = AsyncMock(return_value=("running", SessionResult()))
    for target, value in {
        "get_chat_session": AsyncMock(return_value=sub),
        "stream_registry.get_session": AsyncMock(return_value=None),
        "wait_for_session_result": wait,
        "list_sub_workspace_files": AsyncMock(return_value=[]),
    }.items():
        monkeypatch.setattr(
            f"backend.copilot.tools.get_sub_session_result.{target}", value
        )

    r = await GetSubSessionResultTool()._execute(
        user_id="alice",
        session=MagicMock(session_id="parent", expert_id=None),
        sub_session_id="inner-1",
    )

    assert (r.status, r.question) == ("needs_input", "Which repo?")
    wait.assert_not_awaited()
