"""Tests for find_agent tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.server.v2.chat.tools.find_agent import FindAgentTool
from backend.server.v2.chat.tools.models import AgentCarouselResponse, NoResultsResponse


@pytest.fixture
def find_agent_tool():
    """Create a FindAgentTool instance."""
    return FindAgentTool()


@pytest.fixture
def mock_marketplace_agents():
    """Mock marketplace agents."""
    return [
        MagicMock(
            id="agent-1",
            name="Data Analyzer",
            description="Analyzes data and creates visualizations",
            creator="user123",
            rating=4.5,
            runs=1000,
            category="analytics",
            is_featured=True,
        ),
        MagicMock(
            id="agent-2",
            name="Email Assistant",
            description="Helps manage and send emails",
            creator="user456",
            rating=4.2,
            runs=500,
            category="communication",
            is_featured=False,
        ),
    ]


@pytest.fixture
def mock_library_agents():
    """Mock library agents."""
    return [
        MagicMock(
            graph_id="lib-agent-1",
            graph=MagicMock(
                id="lib-agent-1",
                name="My Custom Agent",
                description="A custom agent for personal use",
            ),
            created_at="2024-01-01T00:00:00Z",
            can_access_graph=True,
        ),
    ]


@pytest.mark.asyncio
async def test_find_agent_no_query_marketplace_only(
    find_agent_tool,
    mock_marketplace_agents,
) -> None:
    """Test finding agents with no query for anonymous user."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        mock_store.search_store_items = AsyncMock(return_value=mock_marketplace_agents)

        result = await find_agent_tool.execute(
            user_id=None,
            session_id="test-session",
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 2
        assert len(result.agents) == 2
        assert result.agents[0].name == "Data Analyzer"
        assert result.agents[0].source == "marketplace"
        assert result.agents[0].is_featured is True


@pytest.mark.asyncio
async def test_find_agent_with_query(find_agent_tool, mock_marketplace_agents) -> None:
    """Test finding agents with search query."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        mock_store.search_store_items = AsyncMock(
            return_value=[mock_marketplace_agents[0]],  # Only returns Data Analyzer
        )

        result = await find_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            query="data analyzer",
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 1
        assert result.agents[0].name == "Data Analyzer"
        assert "data analyzer" in result.message.lower()


@pytest.mark.asyncio
async def test_find_agent_authenticated_with_library(
    find_agent_tool,
    mock_marketplace_agents,
    mock_library_agents,
) -> None:
    """Test finding agents for authenticated user with library agents."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store, patch(
        "backend.server.v2.chat.tools.find_agent.library_db"
    ) as mock_lib:

        mock_store.search_store_items = AsyncMock(return_value=mock_marketplace_agents)
        mock_lib.get_library_agents = AsyncMock(return_value=mock_library_agents)

        result = await find_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 3  # 2 marketplace + 1 library

        # Check that library agent is first
        assert result.agents[0].name == "My Custom Agent"
        assert result.agents[0].source == "library"
        assert result.agents[0].in_library is True

        # Check marketplace agents
        assert result.agents[1].source == "marketplace"
        assert result.agents[1].in_library is True  # Should be marked as in library


@pytest.mark.asyncio
async def test_find_agent_no_results(find_agent_tool) -> None:
    """Test when no agents are found."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        mock_store.search_store_items = AsyncMock(return_value=[])

        result = await find_agent_tool.execute(
            user_id=None,
            session_id="test-session",
            query="nonexistent agent",
        )

        assert isinstance(result, NoResultsResponse)
        assert "nonexistent agent" in result.message.lower()
        assert len(result.suggestions) > 0


@pytest.mark.asyncio
async def test_find_agent_category_filter(
    find_agent_tool, mock_marketplace_agents
) -> None:
    """Test finding agents with category filter."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        # Return only analytics category
        mock_store.search_store_items = AsyncMock(
            return_value=[mock_marketplace_agents[0]],
        )

        result = await find_agent_tool.execute(
            user_id=None,
            session_id="test-session",
            category="analytics",
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 1
        assert result.agents[0].category == "analytics"
        assert "analytics" in result.message.lower()


@pytest.mark.asyncio
async def test_find_agent_featured_only(
    find_agent_tool, mock_marketplace_agents
) -> None:
    """Test finding only featured agents."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        # Return only featured agents
        mock_store.search_store_items = AsyncMock(
            return_value=[mock_marketplace_agents[0]],
        )

        result = await find_agent_tool.execute(
            user_id=None,
            session_id="test-session",
            featured_only=True,
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 1
        assert result.agents[0].is_featured is True
        assert "featured" in result.message.lower()


@pytest.mark.asyncio
async def test_find_agent_with_limit(find_agent_tool, mock_marketplace_agents) -> None:
    """Test limiting number of results."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        mock_store.search_store_items = AsyncMock(
            return_value=mock_marketplace_agents[:1],  # Simulate limit applied
        )

        result = await find_agent_tool.execute(
            user_id=None,
            session_id="test-session",
            limit=1,
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 1
        assert len(result.agents) == 1


@pytest.mark.asyncio
async def test_find_agent_duplicate_removal(
    find_agent_tool,
    mock_marketplace_agents,
    mock_library_agents,
) -> None:
    """Test that duplicate agents are properly handled."""
    # Create a library agent that matches a marketplace agent
    duplicate_library_agent = MagicMock(
        graph_id="agent-1",  # Same as marketplace agent-1
        graph=MagicMock(
            id="agent-1",
            name="Data Analyzer",
            description="Analyzes data and creates visualizations",
        ),
        created_at="2024-01-01T00:00:00Z",
        can_access_graph=True,
    )

    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store, patch(
        "backend.server.v2.chat.tools.find_agent.library_db"
    ) as mock_lib:

        mock_store.search_store_items = AsyncMock(return_value=mock_marketplace_agents)
        mock_lib.get_library_agents = AsyncMock(return_value=[duplicate_library_agent])

        result = await find_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
        )

        assert isinstance(result, AgentCarouselResponse)
        # Should have 2 marketplace agents, but duplicate is removed
        assert result.count == 2

        # Verify no duplicates by ID
        agent_ids = [agent.id for agent in result.agents]
        assert len(agent_ids) == len(set(agent_ids))


@pytest.mark.asyncio
async def test_find_agent_error_handling(find_agent_tool) -> None:
    """Test error handling in find_agent."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store:
        mock_store.search_store_items = AsyncMock(
            side_effect=Exception("Database error"),
        )

        result = await find_agent_tool.execute(
            user_id=None,
            session_id="test-session",
        )

        # Should still return a response, possibly empty
        assert isinstance(result, (AgentCarouselResponse, NoResultsResponse))


@pytest.mark.asyncio
async def test_find_agent_library_search_with_query(
    find_agent_tool,
    mock_library_agents,
) -> None:
    """Test that library agents are filtered by query."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store, patch(
        "backend.server.v2.chat.tools.find_agent.library_db"
    ) as mock_lib:

        mock_store.search_store_items = AsyncMock(return_value=[])
        mock_lib.get_library_agents = AsyncMock(return_value=mock_library_agents)

        result = await find_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            query="custom",  # Should match "My Custom Agent"
        )

        assert isinstance(result, AgentCarouselResponse)
        assert result.count == 1
        assert "custom" in result.agents[0].name.lower()


@pytest.mark.asyncio
async def test_find_agent_anonymous_no_library(
    find_agent_tool, mock_library_agents
) -> None:
    """Test that anonymous users don't get library results."""
    with patch("backend.server.v2.chat.tools.find_agent.store_db") as mock_store, patch(
        "backend.server.v2.chat.tools.find_agent.library_db"
    ) as mock_lib:

        mock_store.search_store_items = AsyncMock(return_value=[])
        mock_lib.get_library_agents = AsyncMock(return_value=mock_library_agents)

        result = await find_agent_tool.execute(
            user_id=None,  # Anonymous
            session_id="test-session",
        )

        # Should not call library_db for anonymous users
        mock_lib.get_library_agents.assert_not_called()

        assert isinstance(result, NoResultsResponse)
