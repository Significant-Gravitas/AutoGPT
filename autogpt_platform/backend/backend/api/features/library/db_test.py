from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.enums
import prisma.models
import pytest

from backend.data.db import connect
from backend.data.includes import library_agent_include

from . import db


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
