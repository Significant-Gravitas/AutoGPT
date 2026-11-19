"""A refused tool call tells the model it did not happen (SECRT-3224).

A routine's run asked for ``schedule_routine``, the routine's permissions
refused it, and Otto went on telling the user the routine was running. These
replay that call through both refusal paths and assert what the model reads
back before it writes its reply: the call did not run, it must say so rather
than claim it did, and on a run nobody is watching, where the work can still
be set up.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.context import set_turn_unattended
from backend.copilot.model import ChatSession
from backend.copilot.permissions import (
    CopilotPermissions,
    denied_tool_names,
    routine_disabled_tools,
)
from backend.copilot.sdk.tool_adapter import (
    _make_truncating_wrapper,
    _text_from_mcp_result,
    reset_pending_tool_outputs,
    reset_tool_failure_counters,
    set_execution_context,
)
from backend.copilot.tools import execute_tool
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.routines import ScheduleRoutineTool

USER = "user-refused"
ROUTINE_INPUT = {
    "title": "Weekly report",
    "prompt": "Summarise this week's sales and send it to me.",
    "crons": ["0 9 * * 1"],
}
SET_IT_UP_IN_CHAT = "the next time the user chats with you"


def _assert_says_it_did_not_happen(text: str) -> None:
    assert "schedule_routine" in text
    assert "did not run" in text
    assert "Tell the user it did not happen" in text
    assert "never say it did" in text


@pytest.fixture
def routine_run():
    """A routine's run on the SDK engine: nobody watching, the permissions the
    scheduler hands it, hidden from the MCP server the way the SDK service
    hides them."""
    session = ChatSession.new(USER, dry_run=False)
    permissions = CopilotPermissions(tools=sorted(routine_disabled_tools(granted=True)))
    set_execution_context(
        USER,
        session,
        permissions=permissions,
        hidden_tools=denied_tool_names(permissions),
    )
    set_turn_unattended(session, scheduled=True)
    reset_pending_tool_outputs()
    reset_tool_failure_counters()
    yield session
    set_turn_unattended(session, scheduled=False)
    set_execution_context(None, None)


async def _dispatcher_must_not_run(args):
    raise AssertionError("the dispatch resolves to schedule_routine itself")


@pytest.mark.asyncio
async def test_a_routine_run_refused_schedule_routine_is_told_it_did_not_happen(
    routine_run,
):
    run_capability = _make_truncating_wrapper(
        _dispatcher_must_not_run, "run_capability", required_args=["id"]
    )

    with patch.object(ScheduleRoutineTool, "_execute", new_callable=AsyncMock) as ran:
        result = await run_capability(
            {"id": "tool:schedule_routine", "input": ROUTINE_INPUT}
        )

    ran.assert_not_awaited()
    assert result.get("isError") is True
    text = _text_from_mcp_result(result)
    assert "tool_disabled" in text
    _assert_says_it_did_not_happen(text)
    assert SET_IT_UP_IN_CHAT in text
    assert routine_run.has_tool_been_called("schedule_routine") is False


@pytest.mark.asyncio
async def test_a_watched_chat_naming_a_refused_tool_is_told_it_did_not_happen():
    """The engine-level refusal used to say only that the tool was missing.
    Someone is in this chat, so there is no later chat to point to."""
    result = await execute_tool(
        tool_name="schedule_routine",
        parameters=ROUTINE_INPUT,
        user_id=USER,
        session=ChatSession.new(USER, dry_run=False),
        tool_call_id="call-refused",
        disabled_groups=[],
        disabled_tools=routine_disabled_tools(granted=True),
    )

    assert result.success is False
    error = ErrorResponse.model_validate_json(result.output)
    assert error.error == "tool_disabled"
    _assert_says_it_did_not_happen(error.message)
    assert SET_IT_UP_IN_CHAT not in error.message
