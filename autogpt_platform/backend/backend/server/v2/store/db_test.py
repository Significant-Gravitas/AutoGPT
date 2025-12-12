from datetime import datetime

import prisma.enums
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


@pytest.mark.asyncio(loop_scope="session")
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
            is_available=False,
            useForOnboarding=False,
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
        is_available=False,
        useForOnboarding=False,
    )

    # Mock active version agent (what we want to return for active version)
    mock_active_agent = prisma.models.StoreAgent(
        listing_id="test-id",
        storeListingVersionId="active-version-id",
        slug="test-agent",
        agent_name="Test Agent Active",
        agent_video="active_video.mp4",
        agent_image=["active_image.jpg"],
        featured=False,
        creator_username="creator",
        creator_avatar="avatar.jpg",
        sub_heading="Test heading active",
        description="Test description active",
        categories=["test"],
        runs=15,
        rating=4.8,
        versions=["1.0", "2.0"],
        updated_at=datetime.now(),
        is_available=True,
        useForOnboarding=False,
    )

    # Create a mock StoreListing result
    mock_store_listing = mocker.MagicMock()
    mock_store_listing.activeVersionId = "active-version-id"
    mock_store_listing.hasApprovedVersion = True
    mock_store_listing.ActiveVersion = mocker.MagicMock()
    mock_store_listing.ActiveVersion.recommendedScheduleCron = None

    # Mock StoreAgent prisma call - need to handle multiple calls
    mock_store_agent = mocker.patch("prisma.models.StoreAgent.prisma")

    # Set up side_effect to return different results for different calls
    def mock_find_first_side_effect(*args, **kwargs):
        where_clause = kwargs.get("where", {})
        if "storeListingVersionId" in where_clause:
            # Second call for active version
            return mock_active_agent
        else:
            # First call for initial lookup
            return mock_agent

    mock_store_agent.return_value.find_first = mocker.AsyncMock(
        side_effect=mock_find_first_side_effect
    )

    # Mock Profile prisma call
    mock_profile = mocker.MagicMock()
    mock_profile.userId = "user-id-123"
    mock_profile_db = mocker.patch("prisma.models.Profile.prisma")
    mock_profile_db.return_value.find_first = mocker.AsyncMock(
        return_value=mock_profile
    )

    # Mock StoreListing prisma call
    mock_store_listing_db = mocker.patch("prisma.models.StoreListing.prisma")
    mock_store_listing_db.return_value.find_first = mocker.AsyncMock(
        return_value=mock_store_listing
    )

    # Call function
    result = await db.get_store_agent_details("creator", "test-agent")

    # Verify results - should use active version data
    assert result.slug == "test-agent"
    assert result.agent_name == "Test Agent Active"  # From active version
    assert result.active_version_id == "active-version-id"
    assert result.has_approved_version is True
    assert (
        result.store_listing_version_id == "active-version-id"
    )  # Should be active version ID

    # Verify mocks called correctly - now expecting 2 calls
    assert mock_store_agent.return_value.find_first.call_count == 2

    # Check the specific calls
    calls = mock_store_agent.return_value.find_first.call_args_list
    assert calls[0] == mocker.call(
        where={"creator_username": "creator", "slug": "test-agent"}
    )
    assert calls[1] == mocker.call(where={"storeListingVersionId": "active-version-id"})

    mock_store_listing_db.return_value.find_first.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
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


@pytest.mark.asyncio(loop_scope="session")
async def test_create_store_submission(mocker):
    # Mock data
    mock_agent = prisma.models.AgentGraph(
        id="agent-id",
        version=1,
        userId="user-id",
        createdAt=datetime.now(),
        isActive=True,
    )

    mock_listing = prisma.models.StoreListing(
        id="listing-id",
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        isDeleted=False,
        hasApprovedVersion=False,
        slug="test-agent",
        agentGraphId="agent-id",
        agentGraphVersion=1,
        owningUserId="user-id",
        Versions=[
            prisma.models.StoreListingVersion(
                id="version-id",
                agentGraphId="agent-id",
                agentGraphVersion=1,
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
                storeListingId="listing-id",
                submissionStatus=prisma.enums.SubmissionStatus.PENDING,
                isAvailable=True,
            )
        ],
        useForOnboarding=False,
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
    mock_store_listing.return_value.create.assert_called_once()


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
        sorted_by="rating",
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


# Vector search tests


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents_vector_search_mocked(mocker):
    """Test vector search uses embedding service and executes query safely."""
    from backend.integrations.embeddings import EMBEDDING_DIMENSIONS

    # Mock embedding service
    mock_embedding = [0.1] * EMBEDDING_DIMENSIONS
    mock_embedding_service = mocker.MagicMock()
    mock_embedding_service.generate_embedding = mocker.AsyncMock(
        return_value=mock_embedding
    )
    mocker.patch(
        "backend.server.v2.store.db.get_embedding_service",
        mocker.MagicMock(return_value=mock_embedding_service),
    )

    # Mock query_raw_with_schema to return empty results
    mocker.patch(
        "backend.server.v2.store.db.query_raw_with_schema",
        mocker.AsyncMock(side_effect=[[], [{"count": 0}]]),
    )

    # Call function with search query
    result = await db.get_store_agents(search_query="test query")

    # Verify embedding service was called
    mock_embedding_service.generate_embedding.assert_called_once_with("test query")

    # Verify results
    assert isinstance(result.agents, list)
    assert len(result.agents) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents_vector_search_with_results(mocker):
    """Test vector search returns properly formatted results."""
    from backend.integrations.embeddings import EMBEDDING_DIMENSIONS

    # Mock embedding service
    mock_embedding = [0.1] * EMBEDDING_DIMENSIONS
    mock_embedding_service = mocker.MagicMock()
    mock_embedding_service.generate_embedding = mocker.AsyncMock(
        return_value=mock_embedding
    )
    mocker.patch(
        "backend.server.v2.store.db.get_embedding_service",
        mocker.MagicMock(return_value=mock_embedding_service),
    )

    # Mock query results
    mock_agents = [
        {
            "slug": "test-agent",
            "agent_name": "Test Agent",
            "agent_image": ["image.jpg"],
            "creator_username": "creator",
            "creator_avatar": "avatar.jpg",
            "sub_heading": "Test heading",
            "description": "Test description",
            "runs": 10,
            "rating": 4.5,
            "categories": ["test"],
            "featured": False,
            "is_available": True,
            "updated_at": datetime.now(),
            "similarity": 0.95,
        }
    ]
    mock_count = [{"count": 1}]

    mocker.patch(
        "backend.server.v2.store.db.query_raw_with_schema",
        mocker.AsyncMock(side_effect=[mock_agents, mock_count]),
    )

    # Call function with search query
    result = await db.get_store_agents(search_query="test query")

    # Verify results
    assert len(result.agents) == 1
    assert result.agents[0].slug == "test-agent"
    assert result.agents[0].agent_name == "Test Agent"
    assert result.pagination.total_items == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_get_store_agents_vector_search_with_filters(mocker):
    """Test vector search works correctly with additional filters."""
    from backend.integrations.embeddings import EMBEDDING_DIMENSIONS

    # Mock embedding service
    mock_embedding = [0.1] * EMBEDDING_DIMENSIONS
    mock_embedding_service = mocker.MagicMock()
    mock_embedding_service.generate_embedding = mocker.AsyncMock(
        return_value=mock_embedding
    )
    mocker.patch(
        "backend.server.v2.store.db.get_embedding_service",
        mocker.MagicMock(return_value=mock_embedding_service),
    )

    # Mock query_raw_with_schema
    mock_query = mocker.patch(
        "backend.server.v2.store.db.query_raw_with_schema",
        mocker.AsyncMock(side_effect=[[], [{"count": 0}]]),
    )

    # Call function with search query and filters
    await db.get_store_agents(
        search_query="test query",
        featured=True,
        creators=["creator1", "creator2"],
        category="AI",
        sorted_by="rating",
    )

    # Verify query was called with parameterized values
    # First call is the main query, second is count
    assert mock_query.call_count == 2

    # Check that the SQL query includes proper parameterization
    first_call_args = mock_query.call_args_list[0]
    sql_query = first_call_args[0][0]

    # Verify key elements of the query
    assert "embedding <=> $1::vector" in sql_query
    assert "featured = true" in sql_query
    assert "creator_username = ANY($" in sql_query
    assert "= ANY(categories)" in sql_query


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_and_store_embedding_success(mocker):
    """Test that embedding generation and storage works correctly."""
    from backend.integrations.embeddings import EMBEDDING_DIMENSIONS

    # Mock embedding service
    mock_embedding = [0.1] * EMBEDDING_DIMENSIONS
    mock_embedding_service = mocker.MagicMock()
    mock_embedding_service.generate_embedding = mocker.AsyncMock(
        return_value=mock_embedding
    )
    mocker.patch(
        "backend.server.v2.store.db.get_embedding_service",
        mocker.MagicMock(return_value=mock_embedding_service),
    )

    # Mock query_raw_with_schema
    mock_query = mocker.patch(
        "backend.server.v2.store.db.query_raw_with_schema",
        mocker.AsyncMock(return_value=[]),
    )

    # Call the internal function
    await db._generate_and_store_embedding(
        store_listing_version_id="version-123",
        name="Test Agent",
        sub_heading="A test agent",
        description="Does testing",
    )

    # Verify embedding service was called with combined text
    mock_embedding_service.generate_embedding.assert_called_once_with(
        "Test Agent A test agent Does testing"
    )

    # Verify database update was called
    mock_query.assert_called_once()
    call_args = mock_query.call_args
    assert "UPDATE" in call_args[0][0]
    assert "embedding = $1::vector" in call_args[0][0]
    assert call_args[0][2] == "version-123"


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_and_store_embedding_empty_text(mocker):
    """Test that embedding is not generated for empty text."""
    # Mock embedding service
    mock_embedding_service = mocker.MagicMock()
    mock_embedding_service.generate_embedding = mocker.AsyncMock()
    mocker.patch(
        "backend.server.v2.store.db.get_embedding_service",
        mocker.MagicMock(return_value=mock_embedding_service),
    )

    # Mock query_raw_with_schema
    mock_query = mocker.patch(
        "backend.server.v2.store.db.query_raw_with_schema",
        mocker.AsyncMock(return_value=[]),
    )

    # Call with empty fields
    await db._generate_and_store_embedding(
        store_listing_version_id="version-123",
        name="",
        sub_heading="",
        description="",
    )

    # Verify embedding service was NOT called
    mock_embedding_service.generate_embedding.assert_not_called()

    # Verify database was NOT updated
    mock_query.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_and_store_embedding_handles_error(mocker):
    """Test that embedding generation errors don't crash the operation."""
    # Mock embedding service to raise an error
    mock_embedding_service = mocker.MagicMock()
    mock_embedding_service.generate_embedding = mocker.AsyncMock(
        side_effect=Exception("API error")
    )
    mocker.patch(
        "backend.server.v2.store.db.get_embedding_service",
        mocker.MagicMock(return_value=mock_embedding_service),
    )

    # Call should not raise - errors are logged but not propagated
    await db._generate_and_store_embedding(
        store_listing_version_id="version-123",
        name="Test Agent",
        sub_heading="A test agent",
        description="Does testing",
    )

    # Verify embedding service was called (and failed)
    mock_embedding_service.generate_embedding.assert_called_once()
