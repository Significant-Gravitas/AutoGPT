"""A gate "ask" parks the call, ends the run, and the next turn resumes it."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall

from backend.copilot.gate import Decision
from backend.copilot.gate.headline import Headline
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import (
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ResponseType, ToolResponseBase

from .conftest import echo_tool, make_inputs, make_session
from .history import HeldToolCall, pending_tool_calls
from .history_store import decode_history, encode_history
from .resume import route_results
from .runner import run_agent_loop
from .service import _fallback_events
from .state import PaiTurnState


class _SendTool(BaseTool):
    """A registry-shaped tool whose effect the gate parks."""

    def __init__(self) -> None:
        self.ran = False

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Send an email"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"to": {"type": "string"}}}

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        self.ran = True
        return ToolResponseBase(
            type=ResponseType.ERROR, message="sent", session_id=session.session_id
        )


def _calls_send_then_answers():
    async def stream(messages: list[ModelMessage], info):
        if pending_tool_calls(messages) or not any(
            m.kind == "response" for m in messages
        ):
            yield {
                0: DeltaToolCall(
                    name="send_email",
                    json_args='{"to": "a@b.c"}',
                    tool_call_id="call-7",
                )
            }
            return
        returned = [
            part.content
            for m in messages
            if m.kind == "request"
            for part in m.parts
            if isinstance(part, ToolReturnPart)
        ]
        yield f"Tool said: {returned[-1]}"

    return stream


@pytest.mark.asyncio
async def test_gate_ask_holds_the_call_and_ends_the_turn(state):
    tool = _SendTool()
    parked = Decision(
        allowed=False,
        reason="Ask First is on for this chat.",
        review_id="review-1",
        headline=Headline(ask="Send an email"),
    )
    check_action = AsyncMock(return_value=parked)
    with (
        patch("backend.copilot.tools.get_tool", return_value=tool),
        patch("backend.copilot.tools.track_tool_called"),
        patch.object(BaseTool, "_released_read", AsyncMock(return_value=None)),
        patch("backend.copilot.gate.check_action", check_action),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
        ),
    ):
        await run_agent_loop(
            make_inputs(
                _calls_send_then_answers(), state, tools=[echo_tool("send_email")]
            ),
            state,
        )

    # The real gate path ran (and is what persists the held row and card).
    check_action.assert_awaited_once()
    assert check_action.await_args is not None
    assert check_action.await_args.args[:2] == ("send_email", {"to": "a@b.c"})
    assert tool.ran is False
    # The run ended on the held call: no second model request, no fallback text.
    assert [h.review_id for h in state.held] == ["review-1"]
    assert state.held[0].tool_call_id == "call-7"
    assert "review-1" in state.held[0].output
    assert _fallback_events(state) == []
    # The card's tool output went out and its row is written like the baseline's.
    outputs = [e for e in state.emitted if isinstance(e, StreamToolOutputAvailable)]
    assert len(outputs) == 1 and outputs[0].success is False
    assert [m.role for m in state.session_messages] == ["assistant", "tool"]
    # The stored history keeps the call pending for the next turn.
    assert [c.tool_call_id for c in pending_tool_calls(state.messages)] == ["call-7"]


@pytest.mark.asyncio
async def test_next_turn_resumes_with_the_users_answer():
    first = PaiTurnState(make_session(), model="m", routing_source="env")
    tool = _SendTool()
    parked = Decision(allowed=False, reason="needs approval", review_id="review-1")
    with (
        patch("backend.copilot.tools.get_tool", return_value=tool),
        patch("backend.copilot.tools.track_tool_called"),
        patch.object(BaseTool, "_released_read", AsyncMock(return_value=None)),
        patch("backend.copilot.gate.check_action", AsyncMock(return_value=parked)),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
        ),
    ):
        await run_agent_loop(
            make_inputs(
                _calls_send_then_answers(), first, tools=[echo_tool("send_email")]
            ),
            first,
        )
    stored = decode_history(encode_history(first.messages, first.held, watermark=3))
    assert stored is not None

    # The card was answered: the gate's resolve_answered delivered this row.
    answer = PendingMessage(
        content='<held_call_result tool="send_email">sent</held_call_result>',
        metadata={"held_call": {"tool_call_id": "call-7", "review_id": "review-1"}},
    )
    resumption = route_results(stored.messages, stored.held, [answer])
    assert resumption.late_results == []
    assert resumption.deferred is not None
    assert resumption.deferred.calls == {"call-7": answer.content}

    second = PaiTurnState(make_session(), model="m", routing_source="env")
    inputs = make_inputs(
        _calls_send_then_answers(),
        second,
        user_prompt=None,
        history=stored.messages,
        deferred_results=resumption.deferred,
        tools=[echo_tool("send_email")],
    )
    inputs.mapper.resumed_call_ids = resumption.resumed_call_ids
    with patch(
        "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
    ):
        await run_agent_loop(inputs, second)

    assert second.assistant_text == f"Tool said: {answer.content}"
    assert second.held == []
    assert pending_tool_calls(second.messages) == []
    # The resumed call does not open a second card.
    tool_events = (
        StreamToolInputStart,
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
    )
    assert not [
        e
        for e in second.emitted
        if isinstance(e, tool_events) and e.toolCallId == "call-7"
    ]


def test_unanswered_call_resumes_with_the_gates_refusal():
    held = [
        HeldToolCall(
            tool_call_id="call-7",
            tool_name="send_email",
            review_id="review-1",
            output="Held for the user's approval",
        )
    ]
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="send it")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="send_email", args="{}", tool_call_id="call-7")
            ]
        ),
    ]
    late = PendingMessage(
        content="older result",
        metadata={"held_call": {"tool_call_id": "call-1"}},
    )
    resumption = route_results(history, held, [late])
    assert resumption.deferred is not None
    assert resumption.deferred.calls == {"call-7": "Held for the user's approval"}
    assert resumption.late_results == [late]
