"""run_agent inside the copilot executor, where Prisma is never connected.

The copilot executor reaches the database only through the DatabaseManager
RPC client. A scheduled copilot turn fires into a session that carries no
organization when its schedule predates org tagging, which sends run_agent
down the default-team fallback. That lookup has to go through the RPC client
too, or it raises Prisma's "Client is not connected to the query engine" and
the scheduled run fails (SECRT-2799).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import ClientNotConnectedError

from backend.data.execution import ExecutionStatus
from backend.executor.scheduler import GraphExecutionJobInfo

from ._test_data import make_session
from .models import ExecutionStartedResponse
from .run_agent import RunAgentTool


@pytest.fixture
def copilot_executor_db(mocker):
    """Prisma disconnected in-process, the DatabaseManager client answering."""
    mocker.patch("backend.data.db.is_connected", return_value=False)
    disconnected = MagicMock()
    disconnected.orgmember.find_first = AsyncMock(side_effect=ClientNotConnectedError())
    disconnected.team.find_first = AsyncMock(side_effect=ClientNotConnectedError())
    mocker.patch("backend.api.features.orgs.db.prisma", disconnected)

    db_manager = MagicMock()
    db_manager.get_user_default_team = AsyncMock(
        return_value=("personal-org", "personal-team")
    )
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client",
        return_value=db_manager,
    )
    return db_manager


@pytest.fixture
def library_agent(mocker):
    lib = MagicMock(graph_id="graph-1", graph_version=1, id="lib-1")
    lib.name = "Weekly Report"
    mocker.patch(
        "backend.copilot.tools.run_agent.get_or_create_library_agent",
        AsyncMock(return_value=lib),
    )
    return lib


def _graph() -> MagicMock:
    graph = MagicMock(id="graph-1", version=1)
    graph.name = "Weekly Report"
    return graph


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduled_turn_runs_agent_without_prisma(
    mocker, copilot_executor_db, library_agent
):
    session = make_session(user_id="user-1")
    session.metadata.origin = "automation"
    assert session.organization_id is None

    mocker.patch("backend.copilot.tools.run_agent.track_chat_outcome")
    mocker.patch(
        "backend.copilot.tools.run_agent._safe_link_to_chat_share", AsyncMock()
    )
    add = mocker.patch(
        "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
        AsyncMock(return_value=MagicMock(id="exec-1", status=ExecutionStatus.QUEUED)),
    )

    response = await RunAgentTool()._run_agent(
        user_id="user-1",
        session=session,
        graph=_graph(),
        graph_credentials={},
        inputs={},
        dry_run=False,
    )

    assert isinstance(response, ExecutionStartedResponse)
    copilot_executor_db.get_user_default_team.assert_awaited_once_with("user-1")
    assert add.await_args.kwargs["organization_id"] == "personal-org"
    assert add.await_args.kwargs["team_id"] == "personal-team"


@pytest.mark.asyncio(loop_scope="session")
async def test_scheduled_turn_schedules_agent_without_prisma(
    mocker, copilot_executor_db, library_agent
):
    session = make_session(user_id="user-1")
    assert session.organization_id is None

    scheduler = AsyncMock()
    scheduler.add_execution_schedule.return_value = GraphExecutionJobInfo(
        id="job-1",
        name="Weekly",
        next_run_time="",
        timezone="UTC",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        cron="0 10 * * 1",
        input_data={},
    )
    mocker.patch(
        "backend.copilot.tools.run_agent.get_scheduler_client",
        return_value=scheduler,
    )

    await RunAgentTool()._schedule_agent(
        user_id="user-1",
        session=session,
        graph=_graph(),
        graph_credentials={},
        inputs={},
        schedule_name="Weekly",
        cron="0 10 * * 1",
        timezone="UTC",
    )

    copilot_executor_db.get_user_default_team.assert_awaited_once_with("user-1")
    kwargs = scheduler.add_execution_schedule.await_args.kwargs
    assert kwargs["organization_id"] == "personal-org"
    assert kwargs["team_id"] == "personal-team"
