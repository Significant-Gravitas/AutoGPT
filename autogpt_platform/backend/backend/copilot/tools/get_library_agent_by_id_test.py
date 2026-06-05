"""Tests for GetLibraryAgentByIdTool — direct by-id lookup of one library agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.get_library_agent_by_id import GetLibraryAgentByIdTool
from backend.copilot.tools.models import (
    AgentsFoundResponse,
    ErrorResponse,
    NoResultsResponse,
)

from ._test_data import make_session

_TEST_USER_ID = "test-user-get-library-agent"


@pytest.fixture
def tool():
    return GetLibraryAgentByIdTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


def _mock_library_agent(library_id: str, name: str):
    agent = MagicMock()
    agent.id = library_id
    agent.name = name
    agent.description = "an agent"
    agent.creator_name = "test-creator"
    status = MagicMock()
    status.value = "ready"
    agent.status = status
    agent.can_access_graph = True
    agent.has_external_trigger = False
    agent.new_output = False
    agent.graph_id = f"graph-{library_id}"
    agent.graph_version = 1
    agent.input_schema = {}
    agent.output_schema = {}
    return agent


@pytest.mark.asyncio
async def test_resolves_agent_by_id(tool, session):
    """A known id resolves to a single agent via direct lookup."""
    lib_db = MagicMock()
    lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=None)
    lib_db.get_library_agent = AsyncMock(
        return_value=_mock_library_agent("lib-1", "Weather Bot")
    )

    with patch(
        "backend.copilot.tools.agent_search.library_db",
        return_value=lib_db,
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="lib-1",
        )

    assert isinstance(result, AgentsFoundResponse)
    assert result.count == 1
    assert result.agents[0].id == "lib-1"
    assert result.agents[0].name == "Weather Bot"


@pytest.mark.asyncio
async def test_missing_agent_returns_no_results(tool, session):
    """An id that resolves to nothing returns NoResults — never a fuzzy
    name-search fallback."""
    lib_db = MagicMock()
    lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=None)
    lib_db.get_library_agent = AsyncMock(return_value=None)

    with patch(
        "backend.copilot.tools.agent_search.library_db",
        return_value=lib_db,
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="does-not-exist",
        )

    assert isinstance(result, NoResultsResponse)


@pytest.mark.asyncio
async def test_empty_agent_id_returns_error(tool, session):
    """No id provided is an error, not a search."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_id="",
    )

    assert isinstance(result, ErrorResponse)
