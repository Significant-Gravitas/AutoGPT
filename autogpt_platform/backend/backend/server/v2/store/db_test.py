from datetime import datetime

import prisma.errors
import prisma.models
import pytest
from prisma import Prisma

import backend.server.v2.store.db as db
from backend.server.v2.store.model import Profile


@pytest.fixture(autouse=True)
async def setup_prisma():
    # Don't register client if already registered
    try:
        Prisma()
    except prisma.errors.ClientAlreadyRegisteredError:
        pass
    yield


@pytest.mark.asyncio
async def test_get_store_agents(mocker):
    # Mock data
    mock_agents = [
        prisma.models.StoreAgent(
            listing_id="test-id",
            storeListingVersionId="version123",
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
            updated_at=datetime.now(),
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


@pytest.mark.asyncio
async def test_get_store_agent_details(mocker):
    # Mock data
    mock_agent = prisma.models.StoreAgent(
        listing_id="test-id",
        storeListingVersionId="version123",
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
        updated_at=datetime.now(),
    )

    # Mock prisma call
    mock_store_agent = mocker.patch("prisma.models.StoreAgent.prisma")
    mock_store_agent.return_value.find_first = mocker.AsyncMock(return_value=mock_agent)

    # Call function
    result = await db.get_store_agent_details("creator", "test-agent")

    # Verify results
    assert result.slug == "test-agent"
    assert result.agent_name == "Test Agent"

    # Verify mock called correctly
    mock_store_agent.return_value.find_first.assert_called_once_with(
        where={"creator_username": "creator", "slug": "test-agent"}
    )


@pytest.mark.asyncio
async def test_get_store_creator_details(mocker):
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
    result = await db.get_store_creator_details("creator")

    # Verify results
    assert result.username == "creator"
    assert result.name == "Test Creator"
    assert result.description == "Test description"
    assert result.avatar_url == "avatar.jpg"

    # Verify mock called correctly
    mock_creator.return_value.find_unique.assert_called_once_with(
        where={"username": "creator"}
    )


@pytest.mark.asyncio
async def test_create_store_submission(mocker):
    # Mock data
    mock_agent = prisma.models.AgentGraph(
        id="agent-id",
        version=1,
        userId="user-id",
        createdAt=datetime.now(),
        isActive=True,
        isTemplate=False,
    )

    mock_listing = prisma.models.StoreListing(
        id="listing-id",
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        isDeleted=False,
        isApproved=False,
        agentId="agent-id",
        agentVersion=1,
        owningUserId="user-id",
        StoreListingVersions=[
            prisma.models.StoreListingVersion(
                id="version-id",
                agentId="agent-id",
                agentVersion=1,
                slug="test-agent",
                name="Test Agent",
                description="Test description",
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                subHeading="Test heading",
                imageUrls=["image.jpg"],
                categories=["test"],
                isFeatured=False,
                isDeleted=False,
                version=1,
                isAvailable=True,
                isApproved=False,
            )
        ],
    )

    # Mock prisma calls
    mock_agent_graph = mocker.patch("prisma.models.AgentGraph.prisma")
    mock_agent_graph.return_value.find_first = mocker.AsyncMock(return_value=mock_agent)

    mock_store_listing = mocker.patch("prisma.models.StoreListing.prisma")
    mock_store_listing.return_value.find_first = mocker.AsyncMock(return_value=None)
    mock_store_listing.return_value.create = mocker.AsyncMock(return_value=mock_listing)

    # Call function
    result = await db.create_store_submission(
        user_id="user-id",
        agent_id="agent-id",
        agent_version=1,
        slug="test-agent",
        name="Test Agent",
        description="Test description",
    )

    # Verify results
    assert result.name == "Test Agent"
    assert result.description == "Test description"
    assert result.store_listing_version_id == "version-id"

    # Verify mocks called correctly
    mock_agent_graph.return_value.find_first.assert_called_once()
    mock_store_listing.return_value.find_first.assert_called_once()
    mock_store_listing.return_value.create.assert_called_once()


@pytest.mark.asyncio
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
        is_featured=False,
    )

    # Call function
    result = await db.update_profile("user-id", profile)

    # Verify results
    assert result.username == "creator"
    assert result.name == "Test Creator"

    # Verify mocks called correctly
    mock_profile_db.return_value.find_first.assert_called_once()
    mock_profile_db.return_value.update.assert_called_once()


@pytest.mark.asyncio
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
