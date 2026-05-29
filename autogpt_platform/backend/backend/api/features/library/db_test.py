from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.enums
import prisma.models
import pytest

from backend.data.db import connect
from backend.data.includes import library_agent_include

from . import db
from . import model as library_model


@pytest.mark.asyncio
async def test_get_library_agents(mocker):
    # Mock data
    mock_user_created = [
        prisma.models.AgentGraph(
            id="agent1",
            version=1,
            name="Test Agent 1",
            description="Test Description 1",
            userId="test-user",
            isActive=True,
            createdAt=datetime.now(),
        )
    ]

    mock_library_agents = [
        prisma.models.LibraryAgent(
            id="ua1",
            userId="test-user",
            agentGraphId="agent2",
            settings="{}",  # type: ignore
            agentGraphVersion=1,
            isCreatedByUser=False,
            isDeleted=False,
            isArchived=False,
            isHidden=False,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            isFavorite=False,
            useGraphIsActiveVersion=True,
            AgentGraph=prisma.models.AgentGraph(
                id="agent2",
                version=1,
                name="Test Agent 2",
                description="Test Description 2",
                userId="other-user",
                isActive=True,
                createdAt=datetime.now(),
            ),
        )
    ]

    # Mock prisma calls
    mock_agent_graph = mocker.patch("prisma.models.AgentGraph.prisma")
    mock_agent_graph.return_value.find_many = mocker.AsyncMock(
        return_value=mock_user_created
    )

    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_many = mocker.AsyncMock(
        return_value=mock_library_agents
    )
    mock_library_agent.return_value.count = mocker.AsyncMock(return_value=1)

    mocker.patch(
        "backend.api.features.library.db._fetch_execution_counts",
        new=mocker.AsyncMock(return_value={}),
    )

    # Call function
    result = await db.list_library_agents("test-user")

    # Verify results
    assert len(result.agents) == 1
    assert result.agents[0].id == "ua1"
    assert result.agents[0].name == "Test Agent 2"
    assert result.agents[0].description == "Test Description 2"
    assert result.agents[0].graph_id == "agent2"
    assert result.agents[0].graph_version == 1
    assert result.agents[0].can_access_graph is False
    assert result.agents[0].is_latest_version is True
    assert result.pagination.total_items == 1
    assert result.pagination.total_pages == 1
    assert result.pagination.current_page == 1
    assert result.pagination.page_size == 50


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_hidden_arg, expected_in_where",
    [
        (False, False),  # only non-hidden
        (True, True),  # only hidden
        (None, "absent"),  # all (no filter applied)
    ],
)
async def test_list_library_agents_is_hidden_filter(
    mocker, is_hidden_arg, expected_in_where
):
    """Verify the is_hidden tri-state correctly maps to the where clause:
    True/False set isHidden to that value, None omits the key entirely."""
    mock_agent_graph = mocker.patch("prisma.models.AgentGraph.prisma")
    mock_agent_graph.return_value.find_many = mocker.AsyncMock(return_value=[])

    mock_find_many = mocker.AsyncMock(return_value=[])
    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_many = mock_find_many
    mock_library_agent.return_value.count = mocker.AsyncMock(return_value=0)

    mocker.patch(
        "backend.api.features.library.db._fetch_execution_counts",
        new=mocker.AsyncMock(return_value={}),
    )

    await db.list_library_agents("test-user", is_hidden=is_hidden_arg)

    where = mock_find_many.call_args.kwargs["where"]
    if expected_in_where == "absent":
        assert "isHidden" not in where
    else:
        assert where["isHidden"] is expected_in_where


@pytest.mark.asyncio(loop_scope="session")
async def test_add_agent_to_library(mocker):
    await connect()

    # Mock data
    mock_store_listing_data = prisma.models.StoreListingVersion(
        id="version123",
        version=1,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        agentGraphId="agent1",
        agentGraphVersion=1,
        name="Test Agent",
        subHeading="Test Agent Subheading",
        imageUrls=["https://example.com/image.jpg"],
        description="Test Description",
        categories=["test"],
        isFeatured=False,
        isDeleted=False,
        isAvailable=True,
        storeListingId="listing123",
        submissionStatus=prisma.enums.SubmissionStatus.APPROVED,
        AgentGraph=prisma.models.AgentGraph(
            id="agent1",
            version=1,
            name="Test Agent",
            description="Test Description",
            userId="creator",
            isActive=True,
            createdAt=datetime.now(),
        ),
    )

    mock_library_agent_data = prisma.models.LibraryAgent(
        id="ua1",
        userId="test-user",
        agentGraphId=mock_store_listing_data.agentGraphId,
        settings="{}",  # type: ignore
        agentGraphVersion=1,
        isCreatedByUser=False,
        isDeleted=False,
        isArchived=False,
        isHidden=False,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=mock_store_listing_data.AgentGraph,
    )

    # Mock prisma calls
    mock_store_listing_version = mocker.patch(
        "prisma.models.StoreListingVersion.prisma"
    )
    mock_store_listing_version.return_value.find_unique = mocker.AsyncMock(
        return_value=mock_store_listing_data
    )

    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.create = mocker.AsyncMock(
        return_value=mock_library_agent_data
    )

    # Mock graph_db.get_graph function that's called in resolve_graph_for_library
    # (lives in _add_to_library.py after refactor, not db.py)
    mock_graph_db = mocker.patch(
        "backend.api.features.library._add_to_library.graph_db"
    )
    mock_graph_model = mocker.Mock()
    mock_graph_model.id = "agent1"
    mock_graph_model.version = 1
    mock_graph_model.nodes = (
        []
    )  # Empty list so _has_human_in_the_loop_blocks returns False
    mock_graph_db.get_graph = mocker.AsyncMock(return_value=mock_graph_model)

    # Mock the model conversion
    mock_from_db = mocker.patch(
        "backend.api.features.library.model.LibraryAgent.from_db"
    )
    mock_from_db.return_value = mocker.Mock()

    # Call function
    await db.add_store_agent_to_library("version123", "test-user")

    # Verify mocks called correctly
    mock_store_listing_version.return_value.find_unique.assert_called_once_with(
        where={"id": "version123"}, include={"AgentGraph": True}
    )
    # Check that create was called with the expected data including settings
    create_call_args = mock_library_agent.return_value.create.call_args
    assert create_call_args is not None

    # Verify the create data structure
    create_data = create_call_args.kwargs["data"]
    expected_create = {
        "User": {"connect": {"id": "test-user"}},
        "AgentGraph": {"connect": {"graphVersionId": {"id": "agent1", "version": 1}}},
        "isCreatedByUser": False,
        "useGraphIsActiveVersion": False,
    }
    for key, value in expected_create.items():
        assert create_data[key] == value

    # Check that settings field is present and is a SafeJson object
    assert "settings" in create_data
    assert hasattr(create_data["settings"], "__class__")  # Should be a SafeJson object

    # Check include parameter
    assert create_call_args.kwargs["include"] == library_agent_include(
        "test-user", include_nodes=False, include_executions=False
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_add_agent_to_library_not_found(mocker):
    await connect()
    # Mock prisma calls
    mock_store_listing_version = mocker.patch(
        "prisma.models.StoreListingVersion.prisma"
    )
    mock_store_listing_version.return_value.find_unique = mocker.AsyncMock(
        return_value=None
    )

    # Call function and verify exception
    with pytest.raises(db.NotFoundError):
        await db.add_store_agent_to_library("version123", "test-user")

    # Verify mock called correctly
    mock_store_listing_version.return_value.find_unique.assert_called_once_with(
        where={"id": "version123"}, include={"AgentGraph": True}
    )


@pytest.mark.asyncio
async def test_get_library_agent_by_graph_id_excludes_archived(mocker):
    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_first = mocker.AsyncMock(return_value=None)

    result = await db.get_library_agent_by_graph_id("test-user", "agent1", 7)

    assert result is None
    mock_library_agent.return_value.find_first.assert_called_once()
    where = mock_library_agent.return_value.find_first.call_args.kwargs["where"]
    assert where == {
        "agentGraphId": "agent1",
        "userId": "test-user",
        "isDeleted": False,
        "isArchived": False,
        "agentGraphVersion": 7,
    }


@pytest.mark.asyncio
async def test_get_library_agent_by_graph_id_can_include_archived(mocker):
    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_first = mocker.AsyncMock(return_value=None)

    result = await db.get_library_agent_by_graph_id(
        "test-user",
        "agent1",
        7,
        include_archived=True,
    )

    assert result is None
    mock_library_agent.return_value.find_first.assert_called_once()
    where = mock_library_agent.return_value.find_first.call_args.kwargs["where"]
    assert where == {
        "agentGraphId": "agent1",
        "userId": "test-user",
        "isDeleted": False,
        "agentGraphVersion": 7,
    }


@pytest.mark.asyncio
async def test_update_graph_in_library_allows_archived_library_agent(mocker):
    graph = mocker.Mock(id="graph-id")
    existing_version = mocker.Mock(version=1, is_active=True)
    graph_model = mocker.Mock()
    created_graph = mocker.Mock(id="graph-id", version=2, is_active=False)
    current_library_agent = mocker.Mock()
    updated_library_agent = mocker.Mock()

    mocker.patch(
        "backend.api.features.library.db.graph_db.get_graph_all_versions",
        new=mocker.AsyncMock(return_value=[existing_version]),
    )
    mocker.patch(
        "backend.api.features.library.db.graph_db.make_graph_model",
        return_value=graph_model,
    )
    mocker.patch(
        "backend.api.features.library.db.graph_db.create_graph",
        new=mocker.AsyncMock(return_value=created_graph),
    )
    mock_get_library_agent = mocker.patch(
        "backend.api.features.library.db.get_library_agent_by_graph_id",
        new=mocker.AsyncMock(return_value=current_library_agent),
    )
    mock_update_library_agent = mocker.patch(
        "backend.api.features.library.db.update_library_agent_version_and_settings",
        new=mocker.AsyncMock(return_value=updated_library_agent),
    )

    result_graph, result_library_agent = await db.update_graph_in_library(
        graph,
        "test-user",
    )

    assert result_graph is created_graph
    assert result_library_agent is updated_library_agent
    assert graph.version == 2
    graph_model.reassign_ids.assert_called_once_with(
        user_id="test-user", reassign_graph_id=False
    )
    mock_get_library_agent.assert_awaited_once_with(
        "test-user",
        "graph-id",
        include_archived=True,
    )
    mock_update_library_agent.assert_awaited_once_with("test-user", created_graph)


@pytest.mark.asyncio
async def test_create_library_agent_uses_upsert():
    """create_library_agent should use upsert (not create) to handle duplicates."""
    mock_graph = MagicMock()
    mock_graph.id = "graph-1"
    mock_graph.version = 1
    mock_graph.user_id = "user-1"
    mock_graph.nodes = []
    mock_graph.sub_graphs = []

    mock_upserted = MagicMock(name="UpsertedLibraryAgent")

    @asynccontextmanager
    async def fake_tx():
        yield None

    with (
        patch("backend.api.features.library.db.transaction", fake_tx),
        patch("prisma.models.LibraryAgent.prisma") as mock_prisma,
        patch(
            "backend.api.features.library.db.add_generated_agent_image",
            new=AsyncMock(),
        ),
        patch(
            "backend.api.features.library.model.LibraryAgent.from_db",
            return_value=MagicMock(),
        ),
    ):
        mock_prisma.return_value.upsert = AsyncMock(return_value=mock_upserted)

        result = await db.create_library_agent(mock_graph, "user-1")

    assert len(result) == 1
    upsert_call = mock_prisma.return_value.upsert.call_args
    assert upsert_call is not None
    # Verify the upsert where clause uses the composite unique key
    where = upsert_call.kwargs["where"]
    assert "userId_agentGraphId_agentGraphVersion" in where
    # Verify the upsert data has both create and update branches
    data = upsert_call.kwargs["data"]
    assert "create" in data
    assert "update" in data
    # Verify update branch restores soft-deleted/archived agents
    assert data["update"]["isDeleted"] is False
    assert data["update"]["isArchived"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("is_hidden", [True, False])
async def test_create_library_agent_preserves_is_hidden_in_upsert(is_hidden):
    """When create_library_agent is called with is_hidden, both branches
    of the upsert must persist that value — re-adding a soft-deleted
    trigger agent should keep it hidden, and creating a fresh hidden
    agent should land hidden in the DB."""
    mock_graph = MagicMock()
    mock_graph.id = "graph-1"
    mock_graph.version = 1
    mock_graph.user_id = "user-1"
    mock_graph.nodes = []
    mock_graph.sub_graphs = []

    @asynccontextmanager
    async def fake_tx():
        yield None

    with (
        patch("backend.api.features.library.db.transaction", fake_tx),
        patch("prisma.models.LibraryAgent.prisma") as mock_prisma,
        patch(
            "backend.api.features.library.db.add_generated_agent_image",
            new=AsyncMock(),
        ),
        patch(
            "backend.api.features.library.model.LibraryAgent.from_db",
            return_value=MagicMock(),
        ),
    ):
        mock_prisma.return_value.upsert = AsyncMock(return_value=MagicMock())
        await db.create_library_agent(mock_graph, "user-1", is_hidden=is_hidden)

    data = mock_prisma.return_value.upsert.call_args.kwargs["data"]
    assert data["create"]["isHidden"] is is_hidden
    assert data["update"]["isHidden"] is is_hidden


@pytest.mark.asyncio
async def test_list_favorite_library_agents(mocker):
    mock_library_agents = [
        prisma.models.LibraryAgent(
            id="fav1",
            userId="test-user",
            agentGraphId="agent-fav",
            settings="{}",  # type: ignore
            agentGraphVersion=1,
            isCreatedByUser=False,
            isDeleted=False,
            isArchived=False,
            isHidden=False,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            isFavorite=True,
            useGraphIsActiveVersion=True,
            AgentGraph=prisma.models.AgentGraph(
                id="agent-fav",
                version=1,
                name="Favorite Agent",
                description="My Favorite",
                userId="other-user",
                isActive=True,
                createdAt=datetime.now(),
            ),
        )
    ]

    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_many = mocker.AsyncMock(
        return_value=mock_library_agents
    )
    mock_library_agent.return_value.count = mocker.AsyncMock(return_value=1)

    mocker.patch(
        "backend.api.features.library.db._fetch_execution_counts",
        new=mocker.AsyncMock(return_value={"agent-fav": 7}),
    )

    result = await db.list_favorite_library_agents("test-user")

    assert len(result.agents) == 1
    assert result.agents[0].id == "fav1"
    assert result.agents[0].name == "Favorite Agent"
    assert result.agents[0].graph_id == "agent-fav"
    assert result.pagination.total_items == 1
    assert result.pagination.total_pages == 1
    assert result.pagination.current_page == 1
    assert result.pagination.page_size == 50


@pytest.mark.asyncio
async def test_list_library_agents_skips_failed_agent(mocker):
    """Agents that fail parsing should be skipped — covers the except branch."""
    mock_library_agents = [
        prisma.models.LibraryAgent(
            id="ua-bad",
            userId="test-user",
            agentGraphId="agent-bad",
            settings="{}",  # type: ignore
            agentGraphVersion=1,
            isCreatedByUser=False,
            isDeleted=False,
            isArchived=False,
            isHidden=False,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            isFavorite=False,
            useGraphIsActiveVersion=True,
            AgentGraph=prisma.models.AgentGraph(
                id="agent-bad",
                version=1,
                name="Bad Agent",
                description="",
                userId="other-user",
                isActive=True,
                createdAt=datetime.now(),
            ),
        )
    ]

    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_many = mocker.AsyncMock(
        return_value=mock_library_agents
    )
    mock_library_agent.return_value.count = mocker.AsyncMock(return_value=1)

    mocker.patch(
        "backend.api.features.library.db._fetch_execution_counts",
        new=mocker.AsyncMock(return_value={}),
    )
    mocker.patch(
        "backend.api.features.library.model.LibraryAgent.from_db",
        side_effect=Exception("parse error"),
    )

    result = await db.list_library_agents("test-user")

    assert len(result.agents) == 0
    assert result.pagination.total_items == 1


@pytest.mark.asyncio
async def test_list_trigger_agents_filters_by_parent_graph_id(mocker):
    """list_trigger_agents queries hidden LibraryAgents whose graph
    contains an AgentExecutorBlock node whose constantInput.graph_id
    matches the parent's graph_id. Verify the filter is wired up
    correctly — this is the load-bearing logic since the relationship
    is derived (no link table)."""
    parent_agent = library_model.LibraryAgent(
        id="parent-id",
        graph_id="parent-graph-id",
        graph_version=1,
        name="Parent",
        description="The parent",
        image_url=None,
        creator_name="",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        is_hidden=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new=mocker.AsyncMock(return_value=parent_agent),
    )

    trigger_prisma = prisma.models.LibraryAgent(
        id="trig-1",
        userId="test-user",
        agentGraphId="trig-graph-id",
        settings="{}",  # type: ignore
        agentGraphVersion=1,
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        isHidden=True,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=prisma.models.AgentGraph(
            id="trig-graph-id",
            version=1,
            name="Trigger Agent",
            description="Watches for new stuff",
            userId="test-user",
            isActive=True,
            createdAt=datetime.now(),
        ),
    )

    mock_find_many = mocker.AsyncMock(return_value=[trigger_prisma])
    mock_prisma = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_prisma.return_value.find_many = mock_find_many

    result = await db.list_trigger_agents(
        user_id="test-user", library_agent_id="parent-id"
    )

    assert len(result) == 1
    assert result[0].id == "trig-1"
    assert result[0].graph_id == "trig-graph-id"

    # Verify the where clause filters on the parent's graph_id via the
    # AgentExecutorBlock's constantInput JSON path — the core of the
    # derived-relationship query.
    where = mock_find_many.call_args.kwargs["where"]
    assert where["userId"] == "test-user"
    assert where["isHidden"] is True
    assert where["isDeleted"] is False
    assert where["isArchived"] is False
    nodes_some = where["AgentGraph"]["is"]["Nodes"]["some"]
    assert nodes_some["agentBlockId"] == db._AGENT_EXECUTOR_BLOCK_ID
    assert nodes_some["constantInput"]["path"] == ["graph_id"]
    assert nodes_some["constantInput"]["equals"] == prisma.Json("parent-graph-id")


@pytest.mark.asyncio
async def test_list_trigger_agents_skips_get_library_agent_when_parent_graph_id_passed(
    mocker,
):
    """When the caller passes parent_graph_id explicitly, the function
    must skip the redundant get_library_agent lookup and use the
    provided graph_id directly. Regression for a Sentry-flagged double
    lookup from list_agent_triggers."""
    get_lib_spy = mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new=mocker.AsyncMock(),
    )
    mock_find_many = mocker.AsyncMock(return_value=[])
    mock_prisma = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_prisma.return_value.find_many = mock_find_many
    mocker.patch(
        "backend.api.features.library.db._fetch_schedule_info",
        new=mocker.AsyncMock(return_value={}),
    )

    result = await db.list_trigger_agents(
        user_id="test-user",
        library_agent_id="parent-id",
        parent_graph_id="explicit-graph-id",
    )

    assert result == []
    get_lib_spy.assert_not_called()
    nodes_some = mock_find_many.call_args.kwargs["where"]["AgentGraph"]["is"]["Nodes"][
        "some"
    ]
    assert nodes_some["constantInput"]["equals"] == prisma.Json("explicit-graph-id")


@pytest.mark.asyncio
async def test_list_trigger_agents_propagates_schedule_info(mocker):
    """Trigger agents have schedules — _fetch_schedule_info must be
    called and its result threaded through LibraryAgent.from_db so the
    `is_scheduled` / `next_scheduled_run` fields are accurate.
    Regression for a Sentry finding."""
    parent_agent = library_model.LibraryAgent(
        id="parent-id",
        graph_id="parent-graph-id",
        graph_version=1,
        name="Parent",
        description="",
        image_url=None,
        creator_name="",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        is_hidden=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new=mocker.AsyncMock(return_value=parent_agent),
    )

    trigger_prisma = prisma.models.LibraryAgent(
        id="trig-1",
        userId="test-user",
        agentGraphId="trig-graph-id",
        settings="{}",  # type: ignore
        agentGraphVersion=1,
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        isHidden=True,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=prisma.models.AgentGraph(
            id="trig-graph-id",
            version=1,
            name="Trigger Agent",
            description="",
            userId="test-user",
            isActive=True,
            createdAt=datetime.now(),
        ),
    )
    mock_prisma = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_prisma.return_value.find_many = mocker.AsyncMock(return_value=[trigger_prisma])

    schedule_info = {"trig-graph-id": "2026-05-01T08:00:00+00:00"}
    schedule_spy = mocker.patch(
        "backend.api.features.library.db._fetch_schedule_info",
        new=mocker.AsyncMock(return_value=schedule_info),
    )

    result = await db.list_trigger_agents(
        user_id="test-user", library_agent_id="parent-id"
    )

    schedule_spy.assert_awaited_once_with("test-user")
    assert len(result) == 1
    assert result[0].is_scheduled is True
    assert result[0].next_scheduled_run is not None


@pytest.mark.asyncio
async def test_list_trigger_agents_returns_empty_when_no_triggers(mocker):
    """When the parent has no trigger agents, return an empty list
    rather than raising — the Triggers tab hides itself in that case."""
    parent_agent = library_model.LibraryAgent(
        id="parent-id",
        graph_id="parent-graph-id",
        graph_version=1,
        name="Parent",
        description="",
        image_url=None,
        creator_name="",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        is_hidden=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new=mocker.AsyncMock(return_value=parent_agent),
    )

    mock_prisma = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_prisma.return_value.find_many = mocker.AsyncMock(return_value=[])

    result = await db.list_trigger_agents(
        user_id="test-user", library_agent_id="parent-id"
    )
    assert result == []


@pytest.mark.asyncio
async def test_fetch_execution_counts_empty_graph_ids():
    result = await db._fetch_execution_counts("user-1", [])
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_execution_counts_uses_group_by(mocker):
    mock_prisma = mocker.patch("prisma.models.AgentGraphExecution.prisma")
    mock_prisma.return_value.group_by = mocker.AsyncMock(
        return_value=[
            {"agentGraphId": "graph-1", "_count": {"_all": 5}},
            {"agentGraphId": "graph-2", "_count": {"_all": 2}},
        ]
    )

    result = await db._fetch_execution_counts(
        "user-1", ["graph-1", "graph-2", "graph-3"]
    )

    assert result == {"graph-1": 5, "graph-2": 2}
    mock_prisma.return_value.group_by.assert_called_once_with(
        by=["agentGraphId"],
        where={
            "userId": "user-1",
            "agentGraphId": {"in": ["graph-1", "graph-2", "graph-3"]},
            "isDeleted": False,
        },
        count=True,
    )
