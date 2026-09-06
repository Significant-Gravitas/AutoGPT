"""Baseline tool names stream before execution finishes and survive hydration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.baseline import service
from backend.copilot.baseline.service import (
    _baseline_conversation_updater,
    _baseline_tool_executor,
    _BaselineStreamState,
    _begin_baseline_tool_round,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sharing.models import sanitize_chat_message
from backend.copilot.tool_display import emit_tool_display_name
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util.prompt import CompressResult
from backend.util.tool_call_loop import LLMLoopResponse, LLMToolCall


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_tool_display_streams_and_persists_before_tool_result(fails: bool):
    state = _BaselineStreamState()
    session = ChatSession.new("user-1", dry_run=False)
    call = LLMToolCall(
        id="call-1", name="run_agent", arguments='{"library_agent_id":"id-1"}'
    )
    response = LLMLoopResponse(response_text="", tool_calls=[call], raw_response=None)
    _begin_baseline_tool_round(state, response)

    async def execute(**kwargs):
        emit_tool_display_name("Daily report")
        assert state.session_messages[0].tool_calls[0]["display_name"] == "Daily report"
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
        response,
        [result],
        transcript_builder=TranscriptBuilder(),
        state=state,
    )
    assistant = state.session_messages[0]
    assert assistant.tool_calls[0]["display_name"] == "Daily report"
    restored = ChatMessage.model_validate_json(assistant.model_dump_json())
    assert (
        sanitize_chat_message(restored).tool_calls[0]["display_name"] == "Daily report"
    )


@pytest.mark.asyncio
async def test_compression_receives_clean_tool_calls_without_mutating_saved_names():
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "run_agent", "arguments": '{"library_agent_id":"id-1"}'},
    }
    message = ChatMessage(
        role="assistant", tool_calls=[{**call, "display_name": "Daily report"}]
    )
    compress = AsyncMock(
        return_value=CompressResult(messages=[], token_count=10, was_compacted=False)
    )
    with (
        patch("backend.copilot.baseline.service.compress_context", new=compress),
        patch(
            "backend.copilot.baseline.service._get_main_client",
            return_value=MagicMock(),
        ),
    ):
        await service._compress_session_messages(
            [message], model="anthropic/claude-sonnet-4-6"
        )

    compress.assert_awaited_once()
    assert compress.await_args is not None
    assert compress.await_args.kwargs["messages"][0]["tool_calls"] == [call]
    assert message.tool_calls == [{**call, "display_name": "Daily report"}]
