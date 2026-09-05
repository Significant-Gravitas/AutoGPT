"""Baseline tool names stream before execution finishes and survive hydration."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.baseline import service
from backend.copilot.baseline.service import (
    _baseline_conversation_updater,
    _baseline_tool_executor,
    _BaselineStreamState,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sharing.models import sanitize_chat_message
from backend.copilot.tool_display import emit_tool_display_name
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util.tool_call_loop import LLMLoopResponse, LLMToolCall


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_tool_display_streams_and_persists_before_tool_result(fails: bool):
    state = _BaselineStreamState()
    session = ChatSession.new("user-1", dry_run=False)
    call = LLMToolCall(
        id="call-1", name="run_agent", arguments='{"library_agent_id":"id-1"}'
    )

    async def execute(**kwargs):
        emit_tool_display_name("Daily report")
        event = state.emitted_events[-1]
        assert event.model_dump(mode="json") == {
            "type": "data-tool-display",
            "id": "call-1",
            "data": {"toolCallId": "call-1", "displayName": "Daily report"},
        }
        if fails:
            raise RuntimeError("execution failed")
        return StreamToolOutputAvailable(
            toolCallId=kwargs["tool_call_id"], output="done"
        )

    with patch(
        "backend.copilot.baseline.service.execute_tool",
        new=AsyncMock(side_effect=execute),
    ):
        result = await _baseline_tool_executor(
            call, [], state=state, user_id="user-1", session=session, disabled_groups=[]
        )
    messages: list[dict] = []
    _baseline_conversation_updater(
        messages,
        LLMLoopResponse(response_text="", tool_calls=[call], raw_response=None),
        [result],
        transcript_builder=TranscriptBuilder(),
        state=state,
    )
    assistant = state.session_messages[0]
    assert assistant.tool_calls[0]["display_name"] == "Daily report"
    assert "display_name" not in messages[0]["tool_calls"][0]
    restored = ChatMessage.model_validate_json(assistant.model_dump_json())
    assert (
        sanitize_chat_message(restored).tool_calls[0]["display_name"] == "Daily report"
    )


@pytest.mark.asyncio
async def test_cancelled_named_call_preserves_completed_parallel_sibling():
    state = _BaselineStreamState()
    session = ChatSession.new("user-1", dry_run=False)
    calls = [
        LLMToolCall(id=call_id, name="run_agent", arguments="{}")
        for call_id in ("pending", "completed")
    ]
    response = LLMLoopResponse(
        response_text="Running both", tool_calls=calls, raw_response=None
    )
    service._begin_baseline_tool_round(state, response)
    state.session_messages[0].sequence = 4
    started = asyncio.Event()

    async def execute(**kwargs):
        call_id = kwargs["tool_call_id"]
        emit_tool_display_name(f"Workflow {call_id}")
        if call_id == "pending":
            started.set()
            await asyncio.Future()
        return StreamToolOutputAvailable(toolCallId=call_id, output="actual result")

    async def execute_call(call):
        return await _baseline_tool_executor(
            call, [], state=state, user_id="user-1", session=session, disabled_groups=[]
        )

    with patch(
        "backend.copilot.baseline.service.execute_tool",
        new=AsyncMock(side_effect=execute),
    ):
        pending = asyncio.create_task(execute_call(calls[0]))
        await started.wait()
        await execute_call(calls[1])
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    state.tool_persistence.finish(state.session_messages)
    state.tool_persistence.finish(state.session_messages)
    assert len(state.session_messages) == 2
    assistant, completed = state.session_messages
    assert [call["display_name"] for call in assistant.tool_calls] == [
        "Workflow pending",
        "Workflow completed",
    ]
    assert assistant.tool_calls_pending_save
    assert completed.tool_call_id == "completed"
    assert completed.content == "actual result"
    assert (
        len(
            [
                event
                for event in state.emitted_events
                if event.type.value == "data-tool-display"
            ]
        )
        == 2
    )
