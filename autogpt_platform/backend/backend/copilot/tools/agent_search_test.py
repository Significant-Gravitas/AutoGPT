"""Tests for agent search direct lookup functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .agent_search import search_agents
from .models import AgentsFoundResponse, NoResultsResponse

_TEST_USER_ID = "test-user-agent-search"


class TestMarketplaceSlugLookup:
    """Tests for creator/slug direct lookup in marketplace search."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_slug_lookup_found(self):
        """creator/slug query returns the agent directly."""
        mock_details = MagicMock()
        mock_details.creator = "testuser"
        mock_details.slug = "my-agent"
        mock_details.agent_name = "My Agent"
        mock_details.description = "A test agent"
        mock_details.rating = 4.5
        mock_details.runs = 100

        mock_store = MagicMock()
        mock_store.get_store_agent_details = AsyncMock(return_value=mock_details)

        with patch(
            "backend.copilot.tools.agent_search.store_db",
            return_value=mock_store,
        ):
            response = await search_agents(
                query="testuser/my-agent",
                source="marketplace",
                session_id="test-session",
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.count == 1
        assert response.agents[0].id == "testuser/my-agent"
        assert response.agents[0].name == "My Agent"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_slug_lookup_not_found_falls_back_to_search(self):
        """creator/slug not found falls back to general search."""
        from backend.util.exceptions import NotFoundError

        mock_store = MagicMock()
        mock_store.get_store_agent_details = AsyncMock(side_effect=NotFoundError(""))

        # Fallback search returns results
        mock_search_results = MagicMock()
        mock_agent = MagicMock()
        mock_agent.creator = "other"
        mock_agent.slug = "similar-agent"
        mock_agent.agent_name = "Similar Agent"
        mock_agent.description = "A similar agent"
        mock_agent.rating = 3.0
        mock_agent.runs = 50
        mock_search_results.agents = [mock_agent]

        mock_store.get_store_agents = AsyncMock(return_value=mock_search_results)

        with patch(
            "backend.copilot.tools.agent_search.store_db",
            return_value=mock_store,
        ):
            response = await search_agents(
                query="testuser/my-agent",
                source="marketplace",
                session_id="test-session",
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.count == 1
        assert response.agents[0].id == "other/similar-agent"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_slug_lookup_not_found_no_search_results(self):
        """creator/slug not found and search returns nothing."""
        from backend.util.exceptions import NotFoundError

        mock_store = MagicMock()
        mock_store.get_store_agent_details = AsyncMock(side_effect=NotFoundError(""))
        mock_search_results = MagicMock()
        mock_search_results.agents = []
        mock_store.get_store_agents = AsyncMock(return_value=mock_search_results)

        with patch(
            "backend.copilot.tools.agent_search.store_db",
            return_value=mock_store,
        ):
            response = await search_agents(
                query="testuser/nonexistent",
                source="marketplace",
                session_id="test-session",
            )

        assert isinstance(response, NoResultsResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_non_slug_query_goes_to_search(self):
        """Regular keyword query skips slug lookup and goes to search."""
        mock_store = MagicMock()
        mock_search_results = MagicMock()
        mock_agent = MagicMock()
        mock_agent.creator = "creator1"
        mock_agent.slug = "email-agent"
        mock_agent.agent_name = "Email Agent"
        mock_agent.description = "Sends emails"
        mock_agent.rating = 4.0
        mock_agent.runs = 200
        mock_search_results.agents = [mock_agent]
        mock_store.get_store_agents = AsyncMock(return_value=mock_search_results)

        with patch(
            "backend.copilot.tools.agent_search.store_db",
            return_value=mock_store,
        ):
            response = await search_agents(
                query="email",
                source="marketplace",
                session_id="test-session",
            )

        assert isinstance(response, AgentsFoundResponse)
        # get_store_agent_details should NOT have been called
        mock_store.get_store_agent_details.assert_not_called()


class TestLibraryUUIDLookup:
    """Tests for UUID direct lookup in library search."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_uuid_lookup_found_by_graph_id(self):
        """UUID query matching a graph_id returns the agent directly."""
        agent_id = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        mock_agent = MagicMock()
        mock_agent.id = "lib-agent-id"
        mock_agent.name = "My Library Agent"
        mock_agent.description = "A library agent"
        mock_agent.creator_name = "testuser"
        mock_agent.status.value = "HEALTHY"
        mock_agent.can_access_graph = True
        mock_agent.has_external_trigger = False
        mock_agent.new_output = False
        mock_agent.graph_id = agent_id
        mock_agent.graph_version = 1
        mock_agent.input_schema = {}
        mock_agent.output_schema = {}

        mock_lib_db = MagicMock()
        mock_lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)

        with patch(
            "backend.copilot.tools.agent_search.library_db",
            return_value=mock_lib_db,
        ):
            response = await search_agents(
                query=agent_id,
                source="library",
                session_id="test-session",
                user_id=_TEST_USER_ID,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.count == 1
        assert response.agents[0].name == "My Library Agent"
