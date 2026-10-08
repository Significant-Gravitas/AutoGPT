"""Pydantic AI run events map onto the baseline's stream wire sequence."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall

from backend.copilot.response_model import (
    StreamFinishStep,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)

from .conftest import make_inputs
from .runner import run_agent_loop


def _tool_then_text():
    calls = {"n": 0}

    async def stream(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {0: DeltaThinkingPart(content="Let me look.")}
            yield "Checking "
            yield "now."
            yield {
                1: DeltaToolCall(
                    name="web_fetch",
                    json_args='{"url": "https://x.test"}',
                    tool_call_id="call-1",
                )
            }
        else:
            yield "Done."

    return stream


@pytest.mark.asyncio
async def test_event_sequence_matches_baseline_shape(state):
    tool_output = StreamToolOutputAvailable(
        toolCallId="call-1", toolName="web_fetch", output='{"ok": true}'
    )
    with (
        patch(
            "backend.copilot.pai.toolset.execute_tool",
            AsyncMock(return_value=tool_output),
        ),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
        ),
    ):
        await run_agent_loop(make_inputs(_tool_then_text(), state), state)

    kinds = [type(e) for e in state.emitted]
    assert kinds == [
        StreamStartStep,
        StreamReasoningStart,
        StreamReasoningDelta,
        StreamReasoningEnd,
        StreamTextStart,
        StreamTextDelta,
        StreamTextDelta,
        StreamTextEnd,
        StreamFinishStep,
        StreamToolInputStart,
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
        StreamStartStep,
        StreamTextStart,
        StreamTextDelta,
        StreamTextEnd,
        StreamFinishStep,
    ]
    tool_input = state.emitted[10]
    assert isinstance(tool_input, StreamToolInputAvailable)
    assert tool_input.input == {"url": "https://x.test"}
    assert state.assistant_text == "Checking now.Done."


@pytest.mark.asyncio
async def test_rows_match_baseline_rows(state):
    tool_output = StreamToolOutputAvailable(
        toolCallId="call-1", toolName="web_fetch", output="page"
    )
    with (
        patch(
            "backend.copilot.pai.toolset.execute_tool",
            AsyncMock(return_value=tool_output),
        ),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
        ),
    ):
        await run_agent_loop(make_inputs(_tool_then_text(), state), state)

    roles = [(m.role, m.content) for m in state.session_messages]
    assert roles == [
        ("reasoning", "Let me look."),
        ("assistant", "Checking now."),
        ("tool", "page"),
    ]
    assistant = state.session_messages[1]
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0]["function"]["name"] == "web_fetch"
    assert assistant.model == "anthropic/claude-test"


@pytest.mark.asyncio
async def test_bad_tool_arguments_close_the_card(state):
    async def stream(messages, info):
        if len(messages) == 1:
            yield {0: DeltaToolCall(name="nope", json_args="{}", tool_call_id="c-9")}
        else:
            yield "ok"

    with patch(
        "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
    ):
        await run_agent_loop(make_inputs(stream, state), state)

    outputs = [e for e in state.emitted if isinstance(e, StreamToolOutputAvailable)]
    assert len(outputs) == 1
    assert outputs[0].toolCallId == "c-9"
    assert outputs[0].success is False
