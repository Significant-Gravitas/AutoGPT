"""Tests for _enrich_agents_with_graph in agent_search module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.agent_search import _enrich_agents_with_graph
from backend.copilot.tools.models import AgentInfo

_TEST_USER_ID = "test-user-agent-search"


def _make_agent(graph_id: str | None = None, graph_version: int = 1) -> AgentInfo:
    return AgentInfo(
        id="agent-1",
        name="Test Agent",
        description="desc",
        source="library",
        graph_id=graph_id,
        graph_version=graph_version,
    )


def _make_fake_graph() -> MagicMock:
    g = MagicMock()
    g.id = "graph-1"
    return g


@pytest.mark.asyncio(loop_scope="session")
async def test_timeout_preserves_successful_fetches():
    """On timeout, agents that already fetched their graph keep the result."""
    fast_agent = _make_agent(graph_id="fast-graph")
    slow_agent = _make_agent(graph_id="slow-graph")
    fake_graph = _make_fake_graph()

    async def mock_get_graph(graph_id, *, version=None, user_id=None, for_export=False):
        if graph_id == "fast-graph":
            return fake_graph
        # Simulate a slow fetch that will be interrupted by the timeout
        await asyncio.sleep(999)
        return _make_fake_graph()

    mock_gdb = MagicMock()
    mock_gdb.get_graph = AsyncMock(side_effect=mock_get_graph)

    with (
        patch("backend.copilot.tools.agent_search.graph_db", return_value=mock_gdb),
        patch("backend.copilot.tools.agent_search._GRAPH_FETCH_TIMEOUT", 0.1),
    ):
        await _enrich_agents_with_graph([fast_agent, slow_agent], _TEST_USER_ID)

    assert (
        fast_agent.graph is fake_graph
    ), "Successfully fetched graph should be preserved after timeout"
    assert (
        slow_agent.graph is None
    ), "Agent that didn't finish fetching should still have graph=None"


@pytest.mark.asyncio(loop_scope="session")
async def test_enrich_success():
    """All agents get their graphs when no timeout occurs."""
    agent = _make_agent(graph_id="g1")
    fake_graph = _make_fake_graph()

    mock_gdb = MagicMock()
    mock_gdb.get_graph = AsyncMock(return_value=fake_graph)

    with patch("backend.copilot.tools.agent_search.graph_db", return_value=mock_gdb):
        result = await _enrich_agents_with_graph([agent], _TEST_USER_ID)

    assert agent.graph is fake_graph
    assert result is None  # no truncation notice for <= _MAX_GRAPH_FETCHES


@pytest.mark.asyncio(loop_scope="session")
async def test_enrich_skips_agents_without_graph_id():
    """Agents without graph_id are not fetched."""
    agent_no_id = _make_agent(graph_id=None)

    mock_gdb = MagicMock()
    mock_gdb.get_graph = AsyncMock()

    with patch("backend.copilot.tools.agent_search.graph_db", return_value=mock_gdb):
        result = await _enrich_agents_with_graph([agent_no_id], _TEST_USER_ID)

    mock_gdb.get_graph.assert_not_called()
    assert result is None
