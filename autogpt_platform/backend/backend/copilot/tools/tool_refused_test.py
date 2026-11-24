"""A refused tool call tells the model it did not happen (SECRT-3224).

A routine's run asked for ``schedule_routine``, the routine's permissions
refused it, and Otto went on telling the user the routine was running. These
replay refused calls through every refusal path and assert what the model
reads back before it writes its reply: the call did not run, it must say so
rather than claim it did, and on a run nobody is watching, where the work can
still be set up. A deferred tool named by mistake is not a refusal, so it is
sent to ``run_capability`` instead.
"""

from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from backend.copilot.context import set_turn_unattended
from backend.copilot.model import ChatSession
from backend.copilot.permissions import (
    CopilotPermissions,
    denied_tool_names,
    routine_disabled_tools,
)
from backend.copilot.sdk.security_hooks import _deny_unlisted
from backend.copilot.sdk.tool_adapter import (
    MCP_TOOL_PREFIX,
    _make_truncating_wrapper,
    _text_from_mcp_result,
    reset_pending_tool_outputs,
    reset_tool_failure_counters,
    set_execution_context,
)
from backend.copilot.tools import execute_tool
from backend.copilot.tools.capability_gates import gate_denied_error
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.routines import ScheduleRoutineTool

USER = "user-refused"
ROUTINE_INPUT = {
    "title": "Weekly report",
    "prompt": "Summarise this week's sales and send it to me.",
    "crons": ["0 9 * * 1"],
}
SET_IT_UP_IN_CHAT = "can be set up from a chat with the user"
DID_NOT_HAPPEN = "Tell the user it did not happen"


def _assert_says_it_did_not_happen(text: str, name: str) -> None:
    assert name in text
    assert "did not run" in text
    assert DID_NOT_HAPPEN in text
    assert "never say it did" in text


@pytest.fixture
def chat():
    """A turn of the user's own chat: someone is watching, nothing denied."""
    session = ChatSession.new(USER, dry_run=False)
    set_execution_context(USER, session)
    set_turn_unattended(session, scheduled=False)
    yield session
    set_execution_context(None, None)


@pytest.fixture
def routine_run():
    """A routine's run: nobody watching, the permissions the scheduler hands
    it, and those tools hidden the way both engines hide them."""
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
    """The incident's path: the SDK engine, ``run_capability`` resolved to the
    tool, and the permission gate refusing it."""
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
    _assert_says_it_did_not_happen(text, "schedule_routine")
    assert SET_IT_UP_IN_CHAT in text
    assert routine_run.has_tool_been_called("schedule_routine") is False


async def _execute(name: str, session: ChatSession, disabled: frozenset[str]):
    result = await execute_tool(
        tool_name=name,
        parameters={},
        user_id=USER,
        session=session,
        tool_call_id="call-refused",
        disabled_groups=[],
        disabled_tools=disabled,
    )
    assert result.success is False
    error = ErrorResponse.model_validate_json(result.output)
    assert error.error == "tool_disabled"
    return error.message


@pytest.mark.asyncio
async def test_a_routine_run_naming_schedule_routine_is_told_it_did_not_happen(
    routine_run,
):
    """The baseline engine's own refusal, on the same run."""
    message = await _execute("schedule_routine", routine_run, frozenset())

    _assert_says_it_did_not_happen(message, "schedule_routine")
    assert SET_IT_UP_IN_CHAT in message


@pytest.mark.parametrize("name", ["web_fetch", "schedule_routine"])
@pytest.mark.asyncio
async def test_a_tool_the_chat_disabled_is_told_it_did_not_happen(chat, name):
    """Only ``disabled_tools`` refuses these, ``web_fetch`` being eager.
    Someone is in the chat, so even a scheduling tool gets no later chat to
    point to."""
    message = await _execute(name, chat, frozenset({name}))

    _assert_says_it_did_not_happen(message, name)
    assert SET_IT_UP_IN_CHAT not in message


@pytest.mark.asyncio
async def test_a_permitted_deferred_tool_named_directly_is_sent_to_run_capability(
    chat,
):
    """A slip, not a denial: telling the model the work cannot be done would
    turn a call it can make into a false "I can't"."""
    message = await _execute("schedule_routine", chat, frozenset())

    assert 'run_capability(id="tool:schedule_routine"' in message
    assert DID_NOT_HAPPEN not in message


@pytest.mark.asyncio
async def test_a_deferred_tool_this_deployment_cannot_run_is_refused(chat):
    """An unavailable tool has no ``tool:`` id, so pointing the model at one
    would only cost it a round trip."""
    with patch.object(
        ScheduleRoutineTool, "is_available", new_callable=PropertyMock
    ) as available:
        available.return_value = False
        message = await _execute("schedule_routine", chat, frozenset())

    assert "run_capability" not in message
    _assert_says_it_did_not_happen(message, "schedule_routine")


def test_describing_a_refused_tool_does_not_say_something_did_not_happen():
    """A lookup attempted nothing, so there is nothing to report as not done."""
    described = gate_denied_error("schedule_routine", "s", describing=True)

    assert "not available" in described.message
    assert DID_NOT_HAPPEN not in described.message


@pytest.mark.parametrize(
    "turn, tool_name, expected",
    [
        ("chat", "web_fetch", "refused"),
        ("chat", "schedule_routine", "slip"),
        ("routine_run", "schedule_routine", "refused, set it up in chat"),
    ],
)
def test_the_sdk_hook_refusal_says_it_did_not_happen_unless_it_was_a_slip(
    request, turn, tool_name, expected
):
    request.getfixturevalue(turn)
    reason = _deny_unlisted(f"{MCP_TOOL_PREFIX}{tool_name}")["hookSpecificOutput"][
        "permissionDecisionReason"
    ]

    if expected == "slip":
        assert "run_capability" in reason and DID_NOT_HAPPEN not in reason
        return
    _assert_says_it_did_not_happen(reason, tool_name)
    assert "[SECURITY]" in reason
    assert (SET_IT_UP_IN_CHAT in reason) == expected.endswith("in chat")
