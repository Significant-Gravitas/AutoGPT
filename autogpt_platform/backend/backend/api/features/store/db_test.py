from datetime import datetime

import prisma.enums
import prisma.errors
import prisma.models
import pytest
from prisma import Prisma

from . import db
from .model import Profile


@pytest.fixture(autouse=True)
async def setup_prisma():
    # Don't register client if already registered
    try:
        Prisma()
    except prisma.errors.ClientAlreadyRegisteredError:
        pass
    yield


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents(mocker):
    # Mock data
    mock_agents = [
        prisma.models.StoreAgent(
            listing_id="test-id",
            listing_version_id="version123",
            slug="test-agent",
            agent_name="Test Agent",
            agent_video=None,
            agent_image=["image.jpg"],
            featured=False,
            creator_username="creator",
            creator_avatar="avatar.jpg",
            sub_heading="Test heading",
            description="Test description",
            categories=[],
            runs=10,
            rating=4.5,
            versions=["1.0"],
            graph_id="test-graph-id",
            graph_versions=["1"],
            updated_at=datetime.now(),
            is_available=False,
            use_for_onboarding=False,
        )
    ]

    # Mock prisma calls
    mock_store_agent = mocker.patch("prisma.models.StoreAgent.prisma")
    mock_store_agent.return_value.find_many = mocker.AsyncMock(return_value=mock_agents)
    mock_store_agent.return_value.count = mocker.AsyncMock(return_value=1)

    # Call function
    result = await db.get_store_agents()

    # Verify results
    assert len(result.agents) == 1
    assert result.agents[0].slug == "test-agent"
    assert result.pagination.total_items == 1

    # Verify mocks called correctly
    mock_store_agent.return_value.find_many.assert_called_once()
    mock_store_agent.return_value.count.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agent_details(mocker):
    # Mock data - StoreAgent view already contains the active version data
    mock_agent = prisma.models.StoreAgent(
        listing_id="test-id",
        listing_version_id="version123",
        slug="test-agent",
        agent_name="Test Agent",
        agent_video="video.mp4",
        agent_image=["image.jpg"],
        featured=False,
        creator_username="creator",
        creator_avatar="avatar.jpg",
        sub_heading="Test heading",
        description="Test description",
        categories=["test"],
        runs=10,
        rating=4.5,
        versions=["1.0"],
        graph_id="test-graph-id",
        graph_versions=["1"],
        updated_at=datetime.now(),
        is_available=True,
        use_for_onboarding=False,
    )

    # Mock StoreAgent prisma call
    mock_store_agent = mocker.patch("prisma.models.StoreAgent.prisma")
    mock_store_agent.return_value.find_first = mocker.AsyncMock(return_value=mock_agent)

    # Call function
    result = await db.get_store_agent_details("creator", "test-agent")

    # Verify results - constructed from the StoreAgent view
    assert result.slug == "test-agent"
    assert result.agent_name == "Test Agent"
    assert result.active_version_id == "version123"
    assert result.has_approved_version is True
    assert result.store_listing_version_id == "version123"
    assert result.graph_id == "test-graph-id"
    assert result.runs == 10
    assert result.rating == 4.5

    # Verify single StoreAgent lookup
    mock_store_agent.return_value.find_first.assert_called_once_with(
        where={"creator_username": "creator", "slug": "test-agent"}
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_creator(mocker):
    # Mock data
    mock_creator_data = prisma.models.Creator(
        name="Test Creator",
        username="creator",
        description="Test description",
        links=["link1"],
        avatar_url="avatar.jpg",
        num_agents=1,
        agent_rating=4.5,
        agent_runs=10,
        top_categories=["test"],
        is_featured=False,
    )

    # Mock prisma call
    mock_creator = mocker.patch("prisma.models.Creator.prisma")
    mock_creator.return_value.find_unique = mocker.AsyncMock()
    # Configure the mock to return values that will pass validation
    mock_creator.return_value.find_unique.return_value = mock_creator_data

    # Call function
    result = await db.get_store_creator("creator")

    # Verify results
    assert result.username == "creator"
    assert result.name == "Test Creator"
    assert result.description == "Test description"
    assert result.avatar_url == "avatar.jpg"

    # Verify mock called correctly
    mock_creator.return_value.find_unique.assert_called_once_with(
        where={"username": "creator"}
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_create_store_submission(mocker):
    now = datetime.now()

    # Mock agent graph (with no pending submissions) and user with profile
    mock_profile = prisma.models.Profile(
        id="profile-id",
        userId="user-id",
        name="Test User",
        username="testuser",
        description="Test",
        isFeatured=False,
        links=[],
        createdAt=now,
        updatedAt=now,
    )
    mock_user = prisma.models.User(
        id="user-id",
        email="test@example.com",
        createdAt=now,
        updatedAt=now,
        Profile=[mock_profile],
        emailVerified=True,
        metadata="{}",  # type: ignore[reportArgumentType]
        integrations="",
        maxEmailsPerDay=1,
        notifyOnAgentRun=True,
        notifyOnZeroBalance=True,
        notifyOnLowBalance=True,
        notifyOnBlockExecutionFailed=True,
        notifyOnContinuousAgentError=True,
        notifyOnDailySummary=True,
        notifyOnWeeklySummary=True,
        notifyOnMonthlySummary=True,
        notifyOnAgentApproved=True,
        notifyOnAgentRejected=True,
        timezone="Europe/Delft",
    )
    mock_agent = prisma.models.AgentGraph(
        id="agent-id",
        version=1,
        userId="user-id",
        createdAt=now,
        isActive=True,
        StoreListingVersions=[],
        User=mock_user,
    )

    # Mock the created StoreListingVersion (returned by create)
    mock_store_listing_obj = prisma.models.StoreListing(
        id="listing-id",
        createdAt=now,
        updatedAt=now,
        isDeleted=False,
        hasApprovedVersion=False,
        slug="test-agent",
        agentGraphId="agent-id",
        owningUserId="user-id",
        useForOnboarding=False,
    )
    mock_version = prisma.models.StoreListingVersion(
        id="version-id",
        agentGraphId="agent-id",
        agentGraphVersion=1,
        name="Test Agent",
        description="Test description",
        createdAt=now,
        updatedAt=now,
        subHeading="",
        imageUrls=[],
        categories=[],
        isFeatured=False,
        isDeleted=False,
        version=1,
        storeListingId="listing-id",
        submissionStatus=prisma.enums.SubmissionStatus.PENDING,
        isAvailable=True,
        submittedAt=now,
        StoreListing=mock_store_listing_obj,
    )

    # Mock prisma calls
    mock_agent_graph = mocker.patch("prisma.models.AgentGraph.prisma")
    mock_agent_graph.return_value.find_first = mocker.AsyncMock(return_value=mock_agent)

    # Mock transaction context manager
    mock_tx = mocker.MagicMock()
    mocker.patch(
        "backend.api.features.store.db.transaction",
        return_value=mocker.AsyncMock(
            __aenter__=mocker.AsyncMock(return_value=mock_tx),
            __aexit__=mocker.AsyncMock(return_value=False),
        ),
    )

    mock_sl = mocker.patch("prisma.models.StoreListing.prisma")
    mock_sl.return_value.find_unique = mocker.AsyncMock(return_value=None)

    mock_slv = mocker.patch("prisma.models.StoreListingVersion.prisma")
    mock_slv.return_value.create = mocker.AsyncMock(return_value=mock_version)

    # Call function
    result = await db.create_store_submission(
        user_id="user-id",
        graph_id="agent-id",
        graph_version=1,
        slug="test-agent",
        name="Test Agent",
        description="Test description",
    )

    # Verify results
    assert result.name == "Test Agent"
    assert result.description == "Test description"
    assert result.listing_version_id == "version-id"

    # Verify mocks called correctly
    mock_agent_graph.return_value.find_first.assert_called_once()
    mock_slv.return_value.create.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_update_profile(mocker):
    # Mock data
    mock_profile = prisma.models.Profile(
        id="profile-id",
        name="Test Creator",
        username="creator",
        userId="user-id",
        description="Test description",
        links=["link1"],
        avatarUrl="avatar.jpg",
        isFeatured=False,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
    )

    # Mock prisma calls
    mock_profile_db = mocker.patch("prisma.models.Profile.prisma")
    mock_profile_db.return_value.find_first = mocker.AsyncMock(
        return_value=mock_profile
    )
    mock_profile_db.return_value.update = mocker.AsyncMock(return_value=mock_profile)

    # Test data
    profile = Profile(
        name="Test Creator",
        username="creator",
        description="Test description",
        links=["link1"],
        avatar_url="avatar.jpg",
    )

    # Call function
    result = await db.update_profile("user-id", profile)

    # Verify results
    assert result.username == "creator"
    assert result.name == "Test Creator"

    # Verify mocks called correctly
    mock_profile_db.return_value.find_first.assert_called_once()
    mock_profile_db.return_value.update.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_get_user_profile(mocker):
    # Mock data
    mock_profile = prisma.models.Profile(
        id="profile-id",
        name="Test User",
        username="testuser",
        description="Test description",
        links=["link1", "link2"],
        avatarUrl="avatar.jpg",
        isFeatured=False,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        userId="user-id",
    )

    # Mock prisma calls
    mock_profile_db = mocker.patch("prisma.models.Profile.prisma")
    mock_profile_db.return_value.find_first = mocker.AsyncMock(
        return_value=mock_profile
    )

    # Call function
    result = await db.get_user_profile("user-id")

    assert result is not None
    # Verify results
    assert result.name == "Test User"
    assert result.username == "testuser"
    assert result.description == "Test description"
    assert result.links == ["link1", "link2"]
    assert result.avatar_url == "avatar.jpg"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents_with_search_parameterized(mocker):
    """Test that search query uses parameterized SQL - validates the fix works"""

    # Call function with search query containing potential SQL injection
    malicious_search = "test'; DROP TABLE StoreAgent; --"
    result = await db.get_store_agents(search_query=malicious_search)

    # Verify query executed safely
    assert isinstance(result.agents, list)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents_with_search_and_filters_parameterized():
    """Test parameterized SQL with multiple filters"""

    # Call with multiple filters including potential injection attempts
    result = await db.get_store_agents(
        search_query="test",
        creators=["creator1'; DROP TABLE Users; --", "creator2"],
        category="AI'; DELETE FROM StoreAgent; --",
        featured=True,
        sorted_by=db.StoreAgentsSortOptions.RATING,
        page=1,
        page_size=20,
    )

    # Verify the query executed without error
    assert isinstance(result.agents, list)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents_search_category_array_injection():
    """Test that category parameter is safely passed as a parameter"""
    # Try SQL injection via category
    malicious_category = "AI'; DROP TABLE StoreAgent; --"
    result = await db.get_store_agents(
        search_query="test",
        category=malicious_category,
    )

    # Verify the query executed without error
    # Category should be parameterized, preventing SQL injection
    assert isinstance(result.agents, list)
