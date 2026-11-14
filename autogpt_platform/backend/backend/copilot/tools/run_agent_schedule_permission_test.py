"""run_agent's scheduling mode answers to the ``schedule_followup`` permission."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.context import set_execution_context
from backend.copilot.permissions import CopilotPermissions, routine_disabled_tools
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.run_agent import RunAgentTool
from backend.copilot.tree import (
    SpawnRequest,
    TurnEnvelope,
    derive_child_envelope,
    root_envelope,
)

from ._test_data import make_session

_USER = "user-1"
_PATH = "backend.copilot.tools.run_agent"
_GRANTED_ROUTINE = CopilotPermissions(
    tools=sorted(routine_disabled_tools(granted=True))
)
_SCHEDULE = {"schedule_name": "Nightly", "cron": "0 3 * * *"}


def _sub_session() -> TurnEnvelope:
    return derive_child_envelope(
        root_envelope("turn-1"), SpawnRequest(shares_memory=True)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "permissions, envelope",
    [(_GRANTED_ROUTINE, None), (None, _sub_session())],
    ids=["granted-routine", "sub-session"],
)
async def test_a_turn_without_schedule_followup_cannot_schedule_a_workflow(
    permissions, envelope
):
    response, schedule, run = await _call(permissions, envelope, **_SCHEDULE)

    assert isinstance(response, ErrorResponse)
    assert response.error == "tool_disabled"
    schedule.assert_not_awaited()
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_granted_routine_still_runs_a_workflow_now():
    _, schedule, run = await _call(_GRANTED_ROUTINE, None)

    run.assert_awaited_once()
    schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_with_schedule_followup_still_schedules_a_workflow():
    _, schedule, _ = await _call(None, None, **_SCHEDULE)

    schedule.assert_awaited_once()


async def _call(
    permissions: CopilotPermissions | None,
    envelope: TurnEnvelope | None,
    **schedule: str,
):
    """Run the tool against a found, runnable workflow, returning the response
    and the scheduling and run steps it would hand off to."""
    session = make_session(_USER)
    graph = MagicMock(id="graph-1", has_external_trigger=False)
    graph.name = "Digest"
    done = ErrorResponse(message="stub", session_id=session.session_id)
    schedule_step = AsyncMock(return_value=done)
    run_step = AsyncMock(return_value=done)
    set_execution_context(_USER, session, permissions=permissions, envelope=envelope)
    try:
        with ExitStack() as stack:
            for name, value in {
                "_bind_builder_graph": AsyncMock(),
                "_agent_graph": AsyncMock(return_value=(graph, MagicMock(id="lib-1"))),
                "require_installed_workflow": AsyncMock(return_value=None),
                "_provider_refusal": MagicMock(return_value=None),
            }.items():
                stack.enter_context(patch(f"{_PATH}.{name}", value))
            for name, value in {
                "_check_prerequisites": AsyncMock(return_value=({}, None)),
                "_schedule_agent": schedule_step,
                "_run_agent": run_step,
            }.items():
                stack.enter_context(patch.object(RunAgentTool, name, value))
            response = await RunAgentTool()._execute(
                user_id=_USER,
                session=session,
                library_agent_id="lib-1",
                use_defaults=True,
                **schedule,
            )
    finally:
        set_execution_context(None, None)
    return response, schedule_step, run_step
