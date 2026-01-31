from datetime import datetime, timedelta, timezone

import prisma.enums
import prisma.models
import pytest

import backend.api.features.store.exceptions
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
    mock_transaction = mocker.patch("backend.api.features.library.db.transaction")
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
    mock_library_agent.return_value.find_unique = mocker.AsyncMock(return_value=None)
    mock_library_agent.return_value.create = mocker.AsyncMock(
        return_value=mock_library_agent_data
    )

    # Mock graph_db.get_graph function that's called to check for HITL blocks
    mock_graph_db = mocker.patch("backend.api.features.library.db.graph_db")
    mock_graph_model = mocker.Mock()
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
    # Check that create was called with the expected data including settings
    create_call_args = mock_library_agent.return_value.create.call_args
    assert create_call_args is not None

    # Verify the main structure
    expected_data = {
        "User": {"connect": {"id": "test-user"}},
        "AgentGraph": {"connect": {"graphVersionId": {"id": "agent1", "version": 1}}},
        "isCreatedByUser": False,
    }

    actual_data = create_call_args[1]["data"]
    # Check that all expected fields are present
    for key, value in expected_data.items():
        assert actual_data[key] == value

    # Check that settings field is present and is a SafeJson object
    assert "settings" in actual_data
    assert hasattr(actual_data["settings"], "__class__")  # Should be a SafeJson object

    # Check include parameter
    assert create_call_args[1]["include"] == library_agent_include(
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
    with pytest.raises(backend.api.features.store.exceptions.AgentNotFoundError):
        await db.add_store_agent_to_library("version123", "test-user")

    # Verify mock called correctly
    mock_store_listing_version.return_value.find_unique.assert_called_once_with(
        where={"id": "version123"}, include={"AgentGraph": True}
    )


@pytest.mark.asyncio
async def test_list_library_agents_sort_by_last_executed(mocker):
    """
    Test LAST_EXECUTED sorting behavior:
    - Agents WITH executions come first, sorted by most recent execution (updatedAt)
    - Agents WITHOUT executions come last, sorted by creation date
    """
    now = datetime.now(timezone.utc)

    # Agent 1: Has execution that finished 1 hour ago
    agent1_execution = prisma.models.AgentGraphExecution(
        id="exec1",
        agentGraphId="agent1",
        agentGraphVersion=1,
        userId="test-user",
        createdAt=now - timedelta(hours=2),
        updatedAt=now - timedelta(hours=1),  # Finished 1 hour ago
        executionStatus=prisma.enums.AgentExecutionStatus.COMPLETED,
        isDeleted=False,
        isShared=False,
    )
    agent1_graph = prisma.models.AgentGraph(
        id="agent1",
        version=1,
        name="Agent With Recent Execution",
        description="Has execution finished 1 hour ago",
        userId="test-user",
        isActive=True,
        createdAt=now - timedelta(days=5),
        Executions=[agent1_execution],
    )
    library_agent1 = prisma.models.LibraryAgent(
        id="lib1",
        userId="test-user",
        agentGraphId="agent1",
        agentGraphVersion=1,
        settings="{}",  # type: ignore
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        createdAt=now - timedelta(days=5),
        updatedAt=now - timedelta(days=5),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=agent1_graph,
    )

    # Agent 2: Has execution that finished 3 hours ago
    agent2_execution = prisma.models.AgentGraphExecution(
        id="exec2",
        agentGraphId="agent2",
        agentGraphVersion=1,
        userId="test-user",
        createdAt=now - timedelta(hours=5),
        updatedAt=now - timedelta(hours=3),  # Finished 3 hours ago
        executionStatus=prisma.enums.AgentExecutionStatus.COMPLETED,
        isDeleted=False,
        isShared=False,
    )
    agent2_graph = prisma.models.AgentGraph(
        id="agent2",
        version=1,
        name="Agent With Older Execution",
        description="Has execution finished 3 hours ago",
        userId="test-user",
        isActive=True,
        createdAt=now - timedelta(days=3),
        Executions=[agent2_execution],
    )
    library_agent2 = prisma.models.LibraryAgent(
        id="lib2",
        userId="test-user",
        agentGraphId="agent2",
        agentGraphVersion=1,
        settings="{}",  # type: ignore
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        createdAt=now - timedelta(days=3),
        updatedAt=now - timedelta(days=3),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=agent2_graph,
    )

    # Agent 3: No executions, created 1 day ago (should come after agents with executions)
    agent3_graph = prisma.models.AgentGraph(
        id="agent3",
        version=1,
        name="Agent Without Executions (Newer)",
        description="No executions, created 1 day ago",
        userId="test-user",
        isActive=True,
        createdAt=now - timedelta(days=1),
        Executions=[],
    )
    library_agent3 = prisma.models.LibraryAgent(
        id="lib3",
        userId="test-user",
        agentGraphId="agent3",
        agentGraphVersion=1,
        settings="{}",  # type: ignore
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        createdAt=now - timedelta(days=1),
        updatedAt=now - timedelta(days=1),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=agent3_graph,
    )

    # Agent 4: No executions, created 2 days ago
    agent4_graph = prisma.models.AgentGraph(
        id="agent4",
        version=1,
        name="Agent Without Executions (Older)",
        description="No executions, created 2 days ago",
        userId="test-user",
        isActive=True,
        createdAt=now - timedelta(days=2),
        Executions=[],
    )
    library_agent4 = prisma.models.LibraryAgent(
        id="lib4",
        userId="test-user",
        agentGraphId="agent4",
        agentGraphVersion=1,
        settings="{}",  # type: ignore
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        createdAt=now - timedelta(days=2),
        updatedAt=now - timedelta(days=2),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=agent4_graph,
    )

    # Return agents in random order to verify sorting works
    mock_library_agents = [
        library_agent3,
        library_agent1,
        library_agent4,
        library_agent2,
    ]

    # Mock prisma calls
    mock_agent_graph = mocker.patch("prisma.models.AgentGraph.prisma")
    mock_agent_graph.return_value.find_many = mocker.AsyncMock(return_value=[])

    mock_library_agent = mocker.patch("prisma.models.LibraryAgent.prisma")
    mock_library_agent.return_value.find_many = mocker.AsyncMock(
        return_value=mock_library_agents
    )

    # Call function with LAST_EXECUTED sort
    result = await db.list_library_agents(
        "test-user",
        sort_by=library_model.LibraryAgentSort.LAST_EXECUTED,
    )

    # Verify sorting order:
    # 1. Agent 1 (execution finished 1 hour ago) - most recent execution
    # 2. Agent 2 (execution finished 3 hours ago) - older execution
    # 3. Agent 3 (no executions, created 1 day ago) - newer creation
    # 4. Agent 4 (no executions, created 2 days ago) - older creation
    assert len(result.agents) == 4
    assert (
        result.agents[0].id == "lib1"
    ), "Agent with most recent execution should be first"
    assert result.agents[1].id == "lib2", "Agent with older execution should be second"
    assert (
        result.agents[2].id == "lib3"
    ), "Agent without executions (newer) should be third"
    assert (
        result.agents[3].id == "lib4"
    ), "Agent without executions (older) should be last"
