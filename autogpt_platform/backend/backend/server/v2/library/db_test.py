from datetime import datetime

import prisma.enums
import prisma.errors
import prisma.models
import prisma.types
import pytest

import backend.server.v2.library.db as db
import backend.server.v2.store.exceptions
from backend.data.db import connect
from backend.data.includes import library_agent_include


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

    # Mock the transaction context
    mock_transaction = mocker.patch("backend.server.v2.library.db.transaction")
    mock_transaction.return_value.__aenter__ = mocker.AsyncMock(return_value=None)
    mock_transaction.return_value.__aexit__ = mocker.AsyncMock(return_value=None)
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
    mock_library_agent.return_value.find_unique = mocker.AsyncMock(return_value=None)
    mock_library_agent.return_value.create = mocker.AsyncMock(
        return_value=mock_library_agent_data
    )

    # Mock the model conversion
    mock_from_db = mocker.patch("backend.server.v2.library.model.LibraryAgent.from_db")
    mock_from_db.return_value = mocker.Mock()

    # Call function
    await db.add_store_agent_to_library("version123", "test-user")

    # Verify mocks called correctly
    mock_store_listing_version.return_value.find_unique.assert_called_once_with(
        where={"id": "version123"}, include={"AgentGraph": True}
    )
    mock_library_agent.return_value.find_unique.assert_called_once_with(
        where={
            "userId_agentGraphId_agentGraphVersion": {
                "userId": "test-user",
                "agentGraphId": "agent1",
                "agentGraphVersion": 1,
            }
        },
        include={"AgentGraph": True},
    )
    mock_library_agent.return_value.create.assert_called_once_with(
        data={
            "User": {"connect": {"id": "test-user"}},
            "AgentGraph": {
                "connect": {"graphVersionId": {"id": "agent1", "version": 1}}
            },
            "isCreatedByUser": False,
        },
        include=library_agent_include(
            "test-user", include_nodes=False, include_executions=False
        ),
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
    with pytest.raises(backend.server.v2.store.exceptions.AgentNotFoundError):
        await db.add_store_agent_to_library("version123", "test-user")

    # Verify mock called correctly
    mock_store_listing_version.return_value.find_unique.assert_called_once_with(
        where={"id": "version123"}, include={"AgentGraph": True}
    )
