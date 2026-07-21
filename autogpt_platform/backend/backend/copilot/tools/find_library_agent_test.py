"""Tests for FindLibraryAgentTool, especially the create-time similarity mode."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.find_library_agent import FindLibraryAgentTool
from backend.copilot.tools.models import (
    AgentsFoundResponse,
    ErrorResponse,
    NoResultsResponse,
)
from backend.util.exceptions import DatabaseError

from ._test_data import make_session

_TEST_USER_ID = "test-user-find-library-agent"


@pytest.fixture
def tool():
    return FindLibraryAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


def _mock_library_agent(library_id: str, name: str, description: str = ""):
    """Build a mock LibraryAgent with the fields _library_agent_to_info reads."""
    agent = MagicMock()
    agent.id = library_id
    agent.name = name
    agent.description = description
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
    agent.trigger_setup_info = None
    return agent


@pytest.mark.asyncio
async def test_for_creation_returns_hybrid_ranked_results(tool, session):
    """for_creation=true delegates to hybrid_search_library_agents and
    sets match_score on each returned AgentInfo (no description mutation)."""
    match_a = {"content_id": "lib-a", "combined_score": 0.91, "relevance": 0.91}
    match_b = {"content_id": "lib-b", "combined_score": 0.72, "relevance": 0.72}

    lib_db = MagicMock()
    lib_db.get_library_agent = AsyncMock(
        side_effect=[
            _mock_library_agent("lib-a", "Email Summariser", "summarises emails"),
            _mock_library_agent("lib-b", "Inbox Triage", "triages inbox"),
        ]
    )

    with (
        patch(
            "backend.copilot.tools.agent_search.hybrid_search_library_agents",
            new=AsyncMock(return_value=[match_a, match_b]),
        ),
        patch(
            "backend.copilot.tools.agent_search.library_db",
            return_value=lib_db,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            for_creation=True,
            goal_summary="summarise my emails",
        )

    assert isinstance(result, AgentsFoundResponse)
    assert result.count == 2
    assert result.agents[0].match_score == 0.91
    assert result.agents[1].match_score == 0.72
    # Description must NOT be mutated with a "[N% match]" prefix anymore.
    assert "match]" not in result.agents[0].description
    assert "match]" not in result.agents[1].description
    # Message should steer the LLM and reference the new field name.
    assert "library_check_ack" in result.message
    assert "match_score" in result.message


@pytest.mark.asyncio
async def test_for_creation_no_matches_returns_no_results(tool, session):
    """When the hybrid search returns nothing, the tool tells the LLM it
    may proceed with create_agent + library_check_ack=true."""
    with patch(
        "backend.copilot.tools.agent_search.hybrid_search_library_agents",
        new=AsyncMock(return_value=[]),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            for_creation=True,
            goal_summary="some niche thing",
        )

    assert isinstance(result, NoResultsResponse)
    assert "library_check_ack" in result.message


@pytest.mark.asyncio
async def test_for_creation_without_goal_summary_soft_fails(tool, session):
    """for_creation=true with no goal_summary returns a NoResultsResponse
    (not an ErrorResponse) so the chat UI doesn't render "Error finding
    agents" — the gate is still satisfied because the tool was called."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        for_creation=True,
    )
    assert isinstance(result, NoResultsResponse)
    assert "goal_summary" in result.message
    assert "library_check_ack" in result.message


@pytest.mark.asyncio
async def test_for_creation_does_not_fall_back_to_query(tool, session):
    """``query`` must NOT substitute for a missing ``goal_summary`` when
    ``for_creation=true``. Falling back would let the search succeed
    while the create_agent gate validator (which only accepts a non-empty
    ``goal_summary``) rejects, forcing the LLM into a retry loop. Sentry
    flagged this MEDIUM-severity misalignment on find_library_agent.py:83.
    """
    with patch(
        "backend.copilot.tools.agent_search.hybrid_search_library_agents",
        new=AsyncMock(return_value=[]),
    ) as mock_search:
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            for_creation=True,
            query="summarise my emails",
        )

    mock_search.assert_not_awaited()
    assert isinstance(result, NoResultsResponse)
    assert "goal_summary" in result.message


@pytest.mark.asyncio
async def test_for_creation_db_error_soft_fails(tool, session):
    """A DatabaseError from the hybrid search degrades to NoResults so the
    UI stays clean and the LLM can proceed."""
    with patch(
        "backend.copilot.tools.agent_search.hybrid_search_library_agents",
        new=AsyncMock(side_effect=DatabaseError("connection refused")),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            for_creation=True,
            goal_summary="email summary agent",
        )
    assert isinstance(result, NoResultsResponse)
    assert "library_check_ack" in result.message


@pytest.mark.asyncio
async def test_for_creation_unexpected_error_soft_fails(tool, session):
    """Any other exception from hybrid search also degrades to NoResults."""
    with patch(
        "backend.copilot.tools.agent_search.hybrid_search_library_agents",
        new=AsyncMock(side_effect=RuntimeError("embedding API down")),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            for_creation=True,
            goal_summary="email summary agent",
        )
    assert isinstance(result, NoResultsResponse)
    assert "library_check_ack" in result.message


@pytest.mark.asyncio
async def test_default_mode_still_uses_substring_search(tool, session):
    """Without for_creation, the tool falls back to the existing
    substring-search path via search_agents."""
    with patch(
        "backend.copilot.tools.find_library_agent.search_agents",
        new=AsyncMock(return_value=NoResultsResponse(message="ok")),
    ) as mock_search:
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            query="email",
        )

    assert isinstance(result, NoResultsResponse)
    mock_search.assert_awaited_once()
    assert mock_search.call_args.kwargs["source"] == "library"
    assert mock_search.call_args.kwargs["query"] == "email"


@pytest.mark.asyncio
async def test_agent_id_resolves_by_direct_lookup(tool, session):
    """agent_id resolves to a single agent via direct by-id lookup."""
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
async def test_agent_id_missing_returns_no_results(tool, session):
    """An agent_id that resolves to nothing returns NoResults — never a fuzzy
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
async def test_agent_id_does_not_fall_back_to_query_search(tool, session):
    """When agent_id is provided, the substring search path is never taken —
    a missing id must not silently degrade into a fuzzy name search."""
    lib_db = MagicMock()
    lib_db.get_library_agent_by_graph_id = AsyncMock(return_value=None)
    lib_db.get_library_agent = AsyncMock(return_value=None)
    list_all = AsyncMock()

    with (
        patch(
            "backend.copilot.tools.agent_search.library_db",
            return_value=lib_db,
        ),
    ):
        lib_db.list_library_agents = list_all
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="does-not-exist",
            query="weather",
        )

    assert isinstance(result, NoResultsResponse)
    list_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_user_returns_error(tool, session):
    """An unauthenticated by-id lookup is an error, not a search."""
    result = await tool._execute(
        user_id=None,
        session=session,
        agent_id="lib-1",
    )

    assert isinstance(result, ErrorResponse)
