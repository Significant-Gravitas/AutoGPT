"""Tests for agent search direct lookup functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .agent_search import _enrich_agents_with_graph, search_agents
from .models import AgentInfo, AgentsFoundResponse, NoResultsResponse

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

    @staticmethod
    def _make_mock_library_agent(
        agent_id: str = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
    ) -> MagicMock:
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
        return mock_agent

    @pytest.mark.asyncio(loop_scope="session")
    async def test_uuid_lookup_found_by_graph_id(self):
        """UUID query matching a graph_id returns the agent directly."""
        agent_id = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        mock_agent = self._make_mock_library_agent(agent_id)

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

    @pytest.mark.asyncio(loop_scope="session")
    async def test_include_graph_fetches_graph(self):
        """include_graph=True attaches BaseGraph to agent results."""
        from backend.data.graph import BaseGraph

        agent_id = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        mock_agent = self._make_mock_library_agent(agent_id)
        mock_lib_db = MagicMock()
        mock_lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)

        fake_graph = BaseGraph(id=agent_id, name="My Library Agent", description="test")
        mock_graph_db = MagicMock()
        mock_graph_db.get_graph = AsyncMock(return_value=fake_graph)

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_graph_db,
            ),
        ):
            response = await search_agents(
                query=agent_id,
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.agents[0].graph is not None
        assert response.agents[0].graph.id == agent_id
        mock_graph_db.get_graph.assert_awaited_once_with(
            agent_id,
            version=1,
            user_id=_TEST_USER_ID,
            for_export=True,
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_include_graph_false_skips_fetch(self):
        """include_graph=False (default) does not fetch graph data."""
        agent_id = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        mock_agent = self._make_mock_library_agent(agent_id)
        mock_lib_db = MagicMock()
        mock_lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)

        mock_graph_db = MagicMock()
        mock_graph_db.get_graph = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_graph_db,
            ),
        ):
            response = await search_agents(
                query=agent_id,
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=False,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.agents[0].graph is None
        mock_graph_db.get_graph.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_include_graph_handles_fetch_failure(self):
        """include_graph=True still returns agents when graph fetch fails."""
        agent_id = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        mock_agent = self._make_mock_library_agent(agent_id)
        mock_lib_db = MagicMock()
        mock_lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)

        mock_graph_db = MagicMock()
        mock_graph_db.get_graph = AsyncMock(side_effect=Exception("DB down"))

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_graph_db,
            ),
        ):
            response = await search_agents(
                query=agent_id,
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.agents[0].graph is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_include_graph_handles_none_return(self):
        """include_graph=True handles get_graph returning None."""
        agent_id = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        mock_agent = self._make_mock_library_agent(agent_id)
        mock_lib_db = MagicMock()
        mock_lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)

        mock_graph_db = MagicMock()
        mock_graph_db.get_graph = AsyncMock(return_value=None)

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_graph_db,
            ),
        ):
            response = await search_agents(
                query=agent_id,
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.agents[0].graph is None


class TestEnrichAgentsWithGraph:
    """Tests for _enrich_agents_with_graph edge cases."""

    @staticmethod
    def _make_mock_library_agent(
        agent_id: str = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
        graph_id: str | None = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
    ) -> MagicMock:
        mock_agent = MagicMock()
        mock_agent.id = f"lib-{agent_id[:8]}"
        mock_agent.name = f"Agent {agent_id[:8]}"
        mock_agent.description = "A library agent"
        mock_agent.creator_name = "testuser"
        mock_agent.status.value = "HEALTHY"
        mock_agent.can_access_graph = True
        mock_agent.has_external_trigger = False
        mock_agent.new_output = False
        mock_agent.graph_id = graph_id
        mock_agent.graph_version = 1
        mock_agent.input_schema = {}
        mock_agent.output_schema = {}
        return mock_agent

    @pytest.mark.asyncio(loop_scope="session")
    async def test_truncation_surfaces_in_response(self):
        """When >_MAX_GRAPH_FETCHES agents have graphs, the response contains a truncation notice."""
        from backend.copilot.tools.agent_search import _MAX_GRAPH_FETCHES
        from backend.data.graph import BaseGraph

        agent_count = _MAX_GRAPH_FETCHES + 5
        mock_agents = []
        for i in range(agent_count):
            uid = f"a1b2c3d4-e5f6-4a7b-8c9d-{i:012d}"
            mock_agents.append(self._make_mock_library_agent(uid, uid))

        mock_lib_db = MagicMock()
        mock_search_results = MagicMock()
        mock_search_results.agents = mock_agents
        mock_lib_db.list_library_agents = AsyncMock(return_value=mock_search_results)

        fake_graph = BaseGraph(id="x", name="g", description="d")
        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock(return_value=fake_graph)

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_gdb,
            ),
        ):
            response = await search_agents(
                query="",
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert mock_gdb.get_graph.await_count == _MAX_GRAPH_FETCHES
        enriched = [a for a in response.agents if a.graph is not None]
        assert len(enriched) == _MAX_GRAPH_FETCHES
        assert "Graph data included for" in response.message
        assert str(_MAX_GRAPH_FETCHES) in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_mixed_graph_id_presence(self):
        """Agents without graph_id are skipped during enrichment."""
        from backend.data.graph import BaseGraph

        agent_with = self._make_mock_library_agent(
            "aaaa0000-0000-0000-0000-000000000001",
            "aaaa0000-0000-0000-0000-000000000001",
        )
        agent_without = self._make_mock_library_agent(
            "bbbb0000-0000-0000-0000-000000000002",
            graph_id=None,
        )

        mock_lib_db = MagicMock()
        mock_search_results = MagicMock()
        mock_search_results.agents = [agent_with, agent_without]
        mock_lib_db.list_library_agents = AsyncMock(return_value=mock_search_results)

        fake_graph = BaseGraph(
            id="aaaa0000-0000-0000-0000-000000000001", name="g", description="d"
        )
        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock(return_value=fake_graph)

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_gdb,
            ),
        ):
            response = await search_agents(
                query="",
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert len(response.agents) == 2
        assert response.agents[0].graph is not None
        assert response.agents[1].graph is None
        mock_gdb.get_graph.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_partial_failure_across_multiple_agents(self):
        """When some graph fetches fail, successful ones still have graphs attached."""
        from backend.data.graph import BaseGraph

        id_ok = "aaaa0000-0000-0000-0000-000000000001"
        id_fail = "bbbb0000-0000-0000-0000-000000000002"
        agent_ok = self._make_mock_library_agent(id_ok, id_ok)
        agent_fail = self._make_mock_library_agent(id_fail, id_fail)

        mock_lib_db = MagicMock()
        mock_search_results = MagicMock()
        mock_search_results.agents = [agent_ok, agent_fail]
        mock_lib_db.list_library_agents = AsyncMock(return_value=mock_search_results)

        fake_graph = BaseGraph(id=id_ok, name="g", description="d")

        async def _side_effect(graph_id, **kwargs):
            if graph_id == id_fail:
                raise Exception("DB error")
            return fake_graph

        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock(side_effect=_side_effect)

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_gdb,
            ),
        ):
            response = await search_agents(
                query="",
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.agents[0].graph is not None
        assert response.agents[0].graph.id == id_ok
        assert response.agents[1].graph is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_keyword_search_with_include_graph(self):
        """include_graph works via keyword search (non-UUID path)."""
        from backend.data.graph import BaseGraph

        agent_id = "cccc0000-0000-0000-0000-000000000003"
        mock_agent = self._make_mock_library_agent(agent_id, agent_id)

        mock_lib_db = MagicMock()
        mock_search_results = MagicMock()
        mock_search_results.agents = [mock_agent]
        mock_lib_db.list_library_agents = AsyncMock(return_value=mock_search_results)

        fake_graph = BaseGraph(id=agent_id, name="g", description="d")
        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock(return_value=fake_graph)

        with (
            patch(
                "backend.copilot.tools.agent_search.library_db",
                return_value=mock_lib_db,
            ),
            patch(
                "backend.copilot.tools.agent_search.graph_db",
                return_value=mock_gdb,
            ),
        ):
            response = await search_agents(
                query="email",
                source="library",
                session_id="s",
                user_id=_TEST_USER_ID,
                include_graph=True,
            )

        assert isinstance(response, AgentsFoundResponse)
        assert response.agents[0].graph is not None
        assert response.agents[0].graph.id == agent_id
        mock_gdb.get_graph.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_timeout_preserves_successful_fetches(self):
        """On timeout, agents that already fetched their graph keep the result."""
        fast_agent = AgentInfo(
            id="a1",
            name="Fast",
            description="d",
            source="library",
            graph_id="fast-graph",
        )
        slow_agent = AgentInfo(
            id="a2",
            name="Slow",
            description="d",
            source="library",
            graph_id="slow-graph",
        )
        fake_graph = MagicMock()
        fake_graph.id = "graph-1"

        async def mock_get_graph(
            graph_id, *, version=None, user_id=None, for_export=False
        ):
            if graph_id == "fast-graph":
                return fake_graph
            await asyncio.sleep(999)
            return MagicMock()

        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock(side_effect=mock_get_graph)

        with (
            patch("backend.copilot.tools.agent_search.graph_db", return_value=mock_gdb),
            patch("backend.copilot.tools.agent_search._GRAPH_FETCH_TIMEOUT", 0.1),
        ):
            await _enrich_agents_with_graph([fast_agent, slow_agent], _TEST_USER_ID)

        assert fast_agent.graph is fake_graph
        assert slow_agent.graph is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_enrich_success(self):
        """All agents get their graphs when no timeout occurs."""
        agent = AgentInfo(
            id="a1", name="Test", description="d", source="library", graph_id="g1"
        )
        fake_graph = MagicMock()
        fake_graph.id = "graph-1"

        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock(return_value=fake_graph)

        with patch(
            "backend.copilot.tools.agent_search.graph_db", return_value=mock_gdb
        ):
            result = await _enrich_agents_with_graph([agent], _TEST_USER_ID)

        assert agent.graph is fake_graph
        assert result is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_enrich_skips_agents_without_graph_id(self):
        """Agents without graph_id are not fetched."""
        agent_no_id = AgentInfo(
            id="a1", name="Test", description="d", source="library", graph_id=None
        )

        mock_gdb = MagicMock()
        mock_gdb.get_graph = AsyncMock()

        with patch(
            "backend.copilot.tools.agent_search.graph_db", return_value=mock_gdb
        ):
            result = await _enrich_agents_with_graph([agent_no_id], _TEST_USER_ID)

        mock_gdb.get_graph.assert_not_called()
        assert result is None
