"""A question for the user ends the baseline turn: nothing runs after it."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import turn_end
from backend.copilot.baseline.service import (
    _baseline_tool_executor,
    _BaselineStreamState,
    _begin_baseline_tool_round,
)
from backend.copilot.baseline.tool_order import runs_alone
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.util.tool_call_loop import LLMLoopResponse, LLMToolCall


@pytest.fixture()
def session():
    session = ChatSession.new("user-1", dry_run=False)
    yield session
    turn_end.clear(session.session_id)


async def _run(call: LLMToolCall, session: ChatSession, execute: AsyncMock):
    state = _BaselineStreamState()
    _begin_baseline_tool_round(
        state, LLMLoopResponse(response_text="", tool_calls=[call], raw_response=None)
    )
    with patch("backend.copilot.baseline.service.execute_tool", new=execute):
        result = await _baseline_tool_executor(
            call,
            [],
            state=state,
            user_id="user-1",
            session=session,
            disabled_groups=[],
            disabled_tools=frozenset(),
        )
    return result, state


@pytest.mark.asyncio
async def test_a_call_after_the_question_is_refused_without_running(session):
    turn_end.request_end(session.session_id)
    execute = AsyncMock()
    call = LLMToolCall(id="c1", name="web_search", arguments='{"query":"x"}')

    result, state = await _run(call, session, execute)

    execute.assert_not_awaited()
    assert result.is_error
    assert result.content == turn_end.ENDED_REFUSAL
    outputs = [
        e for e in state.emitted_events if isinstance(e, StreamToolOutputAvailable)
    ]
    assert outputs[-1].success is False


@pytest.mark.asyncio
async def test_calls_run_normally_before_any_question(session):
    execute = AsyncMock(
        return_value=StreamToolOutputAvailable(toolCallId="c1", output="ok")
    )
    call = LLMToolCall(id="c1", name="web_search", arguments='{"query":"x"}')

    result, _ = await _run(call, session, execute)

    execute.assert_awaited_once()
    assert not result.is_error


def test_the_question_runs_alone_after_the_calls_before_it():
    question = LLMToolCall(id="q", name="ask_question", arguments="{}")
    search = LLMToolCall(id="s", name="web_search", arguments='{"query":"x"}')

    assert runs_alone(question)
    assert not runs_alone(search)


@pytest.mark.asyncio
async def test_after_a_started_task_only_more_tasks_run(session):
    turn_end.request_wrap_up(session.session_id)
    execute = AsyncMock(
        return_value=StreamToolOutputAvailable(toolCallId="c1", output="ok")
    )

    search, _ = await _run(
        LLMToolCall(id="c1", name="web_search", arguments='{"query":"x"}'),
        session,
        execute,
    )
    task, _ = await _run(
        LLMToolCall(id="c2", name="start_task", arguments='{"title":"t","brief":"b"}'),
        session,
        execute,
    )

    assert search.is_error and search.content == turn_end.WRAP_UP_REFUSAL
    assert not task.is_error
    execute.assert_awaited_once()
