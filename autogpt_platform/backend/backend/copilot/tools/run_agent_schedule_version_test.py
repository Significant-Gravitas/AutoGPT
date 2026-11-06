from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.run_agent import RunAgentTool

_PATH = "backend.copilot.tools.run_agent"


@pytest.mark.asyncio
async def test_a_schedule_runs_the_workflow_version_it_was_asked_for():
    """The library entry can point at another version of the workflow; the
    schedule takes the version run_agent resolved and showed."""
    session = ChatSession.new("owner", dry_run=False)
    session.organization_id = "org"
    session.team_id = "team"
    scheduler = MagicMock(
        add_execution_schedule=AsyncMock(side_effect=RuntimeError("captured"))
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
