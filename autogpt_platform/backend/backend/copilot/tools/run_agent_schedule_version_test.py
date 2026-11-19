from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ExecutionStartedResponse
from backend.copilot.tools.run_agent import RunAgentTool
from backend.executor.scheduler import GraphExecutionJobInfo

_PATH = "backend.copilot.tools.run_agent"


@pytest.mark.asyncio
async def test_a_schedule_runs_the_workflow_version_it_was_asked_for():
    """The library entry can point at another version of the workflow; the
    schedule takes the version run_agent resolved and showed."""
    session = ChatSession.new("owner", dry_run=False)
    session.organization_id = "org"
    session.team_id = "team"
    scheduler = MagicMock(
        get_execution_schedules=AsyncMock(return_value=[]),
        add_execution_schedule=AsyncMock(side_effect=RuntimeError("captured")),
    )
    library_agent = MagicMock(graph_id="graph", graph_version=3)
    with (
        patch(
            f"{_PATH}.get_or_create_library_agent",
            AsyncMock(return_value=library_agent),
        ),
        patch(f"{_PATH}.emit_tool_display_name"),
        patch(f"{_PATH}.get_scheduler_client", return_value=scheduler),
    ):
        with pytest.raises(RuntimeError, match="captured"):
            await RunAgentTool()._schedule_agent(
                "owner",
                session,
                MagicMock(id="graph", version=5),
                {},
                {},
                "QA",
                "0 9 * * *",
                "UTC",
            )
    assert scheduler.add_execution_schedule.await_args.kwargs["graph_version"] == 5


@pytest.mark.asyncio
async def test_scheduling_a_workflow_already_scheduled_adds_no_second_one():
    """SECRT-2681: the same workflow, cron, inputs and accounts already have a
    schedule, so the existing one is handed back and nothing is added."""
    session = ChatSession.new("owner", dry_run=False)
    session.organization_id = "org"
    session.team_id = "team"
    existing = GraphExecutionJobInfo(
        schedule_id="noon-check",
        id="noon-check",
        user_id="owner",
        graph_id="graph",
        graph_version=5,
        cron="0 12 * * *",
        input_data={"repo": "AutoGPT"},
        organization_id="org",
        team_id="team",
        name="Trending check",
        next_run_time="2026-09-09T17:00:00+00:00",
        timezone="UTC",
    )
    scheduler = MagicMock(
        get_execution_schedules=AsyncMock(return_value=[existing]),
        add_execution_schedule=AsyncMock(),
    )
    library_agent = MagicMock(graph_id="graph", id="lib", graph_version=5)
    library_agent.name = "Trending"
    with (
        patch(
            f"{_PATH}.get_or_create_library_agent",
            AsyncMock(return_value=library_agent),
        ),
        patch(f"{_PATH}.emit_tool_display_name"),
        patch(f"{_PATH}.get_scheduler_client", return_value=scheduler),
    ):
        result = await RunAgentTool()._schedule_agent(
            "owner",
            session,
            MagicMock(id="graph", version=5),
            {},
            {"repo": "AutoGPT"},
            "Daily trending",
            "0 12 * * *",
            "UTC",
        )
    assert isinstance(result, ExecutionStartedResponse)
    assert result.execution_id == "noon-check"
    assert "no second schedule was added" in result.message
    scheduler.add_execution_schedule.assert_not_awaited()
