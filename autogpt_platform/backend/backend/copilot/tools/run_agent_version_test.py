from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot.model import ChatSession
from backend.copilot.tools.run_agent import RunAgentTool
from backend.data.graph import GraphModel, validate_graph_execution_permissions
from backend.util.exceptions import GraphNotAccessibleError


class ExecutionReachedPersistence(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("library_version", [1, 2])
@pytest.mark.parametrize("readable_versions", [{1}, {1, 2}])
async def test_run_agent_persists_the_validated_version(
    mocker: MockerFixture,
    dry_run: bool,
    library_version: int,
    readable_versions: set[int],
):
    graph, session, database = _execution_setup(
        mocker, library_version, readable_versions
    )

    with pytest.raises(ExecutionReachedPersistence):
        await RunAgentTool()._run_agent(
            user_id="library-user",
            session=session,
            graph=graph,
            graph_credentials={},
            inputs={},
            dry_run=dry_run,
        )

    database.create_graph_execution.assert_awaited_once()
    persisted = database.create_graph_execution.await_args.kwargs
    assert persisted["graph_id"] == graph.id
    assert persisted["graph_version"] == graph.version
    assert persisted["is_dry_run"] is dry_run
    assert persisted["organization_id"] == session.organization_id


@pytest.mark.asyncio
async def test_run_agent_still_denies_an_inaccessible_selected_version(
    mocker: MockerFixture,
):
    graph, session, database = _execution_setup(mocker, 1, set())

    with pytest.raises(GraphNotAccessibleError):
        await RunAgentTool()._run_agent(
            user_id="library-user",
            session=session,
            graph=graph,
            graph_credentials={},
            inputs={},
            dry_run=False,
        )

    database.create_graph_execution.assert_not_awaited()


def _execution_setup(
    mocker: MockerFixture, library_version: int, readable_versions: set[int]
):
    graph = GraphModel(
        id="workflow-id",
        version=1,
        is_active=False,
        user_id="creator",
        created_at=datetime.now(UTC),
        name="Installed workflow",
        description="Version selected before execution",
    )
    session = ChatSession.new("library-user", dry_run=False)
    session.organization_id = "library-org"
    session.team_id = "library-team"
    library = MagicMock(
        id="library-id", graph_id=graph.id, graph_version=library_version
    )
    library.name = graph.name
    mocker.patch(
        "backend.copilot.tools.run_agent.get_or_create_library_agent",
        AsyncMock(return_value=library),
    )
    _mock_version_access(mocker, graph, readable_versions)
    database = _mock_execution_database(mocker, graph)
    return graph, session, database


def _mock_execution_database(mocker: MockerFixture, graph: GraphModel):
    versions = {1: graph, 2: graph.model_copy(update={"version": 2, "is_active": True})}
    database = MagicMock()
    database.get_graph = AsyncMock(
        side_effect=lambda **kwargs: versions[kwargs["version"] or 2]
    )
    database.validate_graph_execution_permissions = validate_graph_execution_permissions
    database.resolve_member_tenancy = AsyncMock(
        side_effect=lambda _user_id, org_id, team_id: (org_id, team_id)
    )
    database.create_graph_execution = AsyncMock(side_effect=ExecutionReachedPersistence)
    mocker.patch("backend.executor.utils.prisma", MagicMock(is_connected=lambda: False))
    mocker.patch(
        "backend.executor.utils.get_database_manager_async_client",
        return_value=database,
    )
    mocker.patch(
        "backend.executor.utils.is_user_paywalled", AsyncMock(return_value=False)
    )
    mocker.patch("backend.executor.utils.is_user_banned", AsyncMock(return_value=False))
    mocker.patch(
        "backend.executor.utils._construct_starting_node_execution_input",
        AsyncMock(return_value=([], set())),
    )
    mocker.patch("backend.executor.utils.record_graph_execution")
    return database


def _mock_version_access(
    mocker: MockerFixture, graph: GraphModel, readable_versions: set[int]
):
    mocker.patch(
        "backend.data.graph.AgentGraph.prisma",
        return_value=MagicMock(
            find_unique=AsyncMock(return_value=MagicMock(userId=graph.user_id))
        ),
    )
    mocker.patch(
        "backend.data.graph.LibraryAgent.prisma",
        return_value=MagicMock(
            find_first=AsyncMock(
                side_effect=lambda *, where: (
                    MagicMock()
                    if where.get("agentGraphVersion", 1) in readable_versions
                    else None
                )
            )
        ),
    )
    mocker.patch(
        "backend.data.graph.is_graph_published_in_marketplace",
        AsyncMock(return_value=False),
    )
