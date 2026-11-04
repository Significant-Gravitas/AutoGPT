"""Every tool call is its own Sentry transaction in its turn's trace, so a
turn's transaction stays bounded however many tools the turn runs."""

from unittest.mock import MagicMock

import pytest
import sentry_sdk

from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse
from backend.util import tracing


def _tool(body) -> BaseTool:
    class _Traced(BaseTool):
        @property
        def name(self) -> str:
            return "read_workspace_file"

        @property
        def description(self) -> str:
            return "traced"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        async def _execute(self, user_id, session, **kwargs):
            return await body()

    return _Traced()


@pytest.mark.asyncio
async def test_a_tool_call_is_a_child_transaction_of_its_turn(sentry_transactions):
    async def body():
        with sentry_sdk.start_span(op="http.client", name="POST /get_graph"):
            return ErrorResponse(message="fine", session_id="s1")

    with tracing.start_transaction("copilot.turn", "copilot.turn") as turn:
        await _tool(body).execute(
            "u1", MagicMock(session_id="s1", external_caller=False), "call-1"
        )

    call, outer = sentry_transactions
    trace = call["contexts"]["trace"]
    assert call["transaction"] == "read_workspace_file"
    assert trace["op"] == "copilot.tool"
    assert trace["trace_id"] == turn.trace_id
    assert trace["parent_span_id"] == turn.span_id
    assert trace["data"]["copilot.tool_call_id"] == "call-1"
    assert call["tags"]["copilot.tool.success"] == "true"
    assert [span["op"] for span in call["spans"]] == ["http.client"]
    assert outer["spans"] == []


@pytest.mark.asyncio
async def test_a_tool_that_raises_fails_its_transaction(sentry_transactions):
    async def body():
        raise RuntimeError("tool crashed")

    result = await _tool(body).execute("u1", MagicMock(session_id="s1"), "call-2")

    assert result.success is False
    [call] = sentry_transactions
    assert call["contexts"]["trace"]["status"] == "internal_error"
    assert call["tags"]["copilot.tool.success"] == "false"
