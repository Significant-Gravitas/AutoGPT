"""Messages queued mid-turn are drained between tool rounds."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall

from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import (
    StreamPendingDrained,
    StreamToolOutputAvailable,
)

from .conftest import make_inputs
from .runner import run_agent_loop


def _two_tool_rounds(seen: list[list[ModelMessage]]):
    async def stream(messages: list[ModelMessage], info):
        seen.append(list(messages))
        responses = sum(1 for m in messages if m.kind == "response")
        if responses < 2:
            yield {
                0: DeltaToolCall(
                    name="web_fetch",
                    json_args='{"url": "https://x.test"}',
                    tool_call_id=f"call-{responses}",
                )
            }
        else:
            yield "All done."

    return stream


def _prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for m in messages
        if m.kind == "request"
        for part in m.parts
        if isinstance(part, UserPromptPart)
    ]


@pytest.mark.asyncio
async def test_follow_up_is_injected_before_the_next_request(state):
    seen: list[list[ModelMessage]] = []
    follow_up = PendingMessage(content="also check y.test")
    drains = AsyncMock(side_effect=[[follow_up], [], []])
    persisted = AsyncMock(return_value=True)
    output = StreamToolOutputAvailable(
        toolCallId="x", toolName="web_fetch", output="page"
    )
    with (
        patch(
            "backend.copilot.pai.toolset.execute_tool", AsyncMock(return_value=output)
        ),
        patch("backend.copilot.pai.runner.drain_pending_safe", drains),
        patch("backend.copilot.pai.runner.persist_pending_as_user_rows", persisted),
        patch(
            "backend.copilot.pai.persistence.persist_session_safe",
            AsyncMock(side_effect=lambda session, prefix: session),
        ),
    ):
        await run_agent_loop(make_inputs(_two_tool_rounds(seen), state), state)

    # Drained only between rounds: not before the first request.
    assert drains.await_count == 2
    first_round, second_round = seen[0], seen[1]
    assert not any("user_follow_up" in p for p in _prompts(first_round))
    follow_ups = [p for p in _prompts(second_round) if "<user_follow_up>" in p]
    assert len(follow_ups) == 1 and "also check y.test" in follow_ups[0]
    # Persisted as its own user row, after the round's rows, like the baseline.
    persisted.assert_awaited_once()
    assert persisted.await_args is not None
    assert persisted.await_args.args[2] == [follow_up]
    rows = [m.role for m in state.session.messages]
    assert rows[-2:] == ["assistant", "tool"]
    drained = [e for e in state.emitted if isinstance(e, StreamPendingDrained)]
    assert len(drained) == 1 and drained[0].messages[0].content == "also check y.test"
    assert state.assistant_text.endswith("All done.")


@pytest.mark.asyncio
async def test_rolled_back_follow_up_is_not_shown_to_the_model(state):
    seen: list[list[ModelMessage]] = []
    output = StreamToolOutputAvailable(toolCallId="x", toolName="web_fetch", output="p")
    with (
        patch(
            "backend.copilot.pai.toolset.execute_tool", AsyncMock(return_value=output)
        ),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe",
            AsyncMock(side_effect=[[PendingMessage(content="later")], [], []]),
        ),
        patch(
            "backend.copilot.pai.runner.persist_pending_as_user_rows",
            AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.pai.persistence.persist_session_safe",
            AsyncMock(side_effect=lambda session, prefix: session),
        ),
    ):
        await run_agent_loop(make_inputs(_two_tool_rounds(seen), state), state)

    assert not any("<user_follow_up>" in p for p in _prompts(seen[-1]))
    assert not [e for e in state.emitted if isinstance(e, StreamPendingDrained)]


@pytest.mark.asyncio
async def test_last_round_hides_tools_and_asks_for_a_summary(state):
    seen: list[list[ModelMessage]] = []
    tool_lists: list[int] = []

    async def stream(messages, info):
        seen.append(list(messages))
        tool_lists.append(len(info.function_tools))
        if info.function_tools:
            yield {
                0: DeltaToolCall(
                    name="web_fetch", json_args="{}", tool_call_id=f"c{len(seen)}"
                )
            }
        else:
            yield "Summary."

    output = StreamToolOutputAvailable(toolCallId="x", toolName="web_fetch", output="p")
    with (
        patch(
            "backend.copilot.pai.toolset.execute_tool", AsyncMock(return_value=output)
        ),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
        ),
    ):
        await run_agent_loop(make_inputs(stream, state, max_rounds=2), state)

    assert tool_lists == [1, 0]
    assert any("tool-call budget" in p for p in _prompts(seen[-1]))
    assert state.budget_reached is True
