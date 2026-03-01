"""Tests for FixAgentGraphTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.fix_agent import FixAgentGraphTool
from backend.copilot.tools.models import ErrorResponse, FixResultResponse

from ._test_data import make_session

_TEST_USER_ID = "test-user-fix-agent"


@pytest.fixture
def tool():
    return FixAgentGraphTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


@pytest.mark.asyncio
async def test_missing_agent_json_returns_error(tool, session):
    """Missing agent_json returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "agent_json" in result.error.lower()


@pytest.mark.asyncio
async def test_empty_nodes_returns_error(tool, session):
    """Agent JSON with no nodes returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_json={"nodes": [], "links": []},
    )
    assert isinstance(result, ErrorResponse)
    assert "no nodes" in result.message.lower()


@pytest.mark.asyncio
async def test_fix_and_validate_success(tool, session):
    """Fixer applies fixes and validator passes -> valid_after_fix=True."""
    agent_json = {
        "nodes": [
            {
                "id": "node-1",
                "block_id": "block-1",
                "input_default": {},
                "metadata": {"position": {"x": 0, "y": 0}},
            }
        ],
        "links": [],
    }

    fixed_agent = dict(agent_json)  # Fixer returns the full agent dict

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = AsyncMock(return_value=fixed_agent)
    mock_fixer.get_fixes_applied.return_value = ["Fixed node UUID format"]

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    with (
        patch(
            "backend.copilot.tools.fix_agent.get_blocks_as_dicts",
            return_value=[],
        ),
        patch(
            "backend.copilot.tools.fix_agent.AgentFixer",
            return_value=mock_fixer,
        ),
        patch(
            "backend.copilot.tools.fix_agent.AgentValidator",
            return_value=mock_validator,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, FixResultResponse)
    assert result.valid_after_fix is True
    assert result.fix_count == 1
    assert result.fixes_applied == ["Fixed node UUID format"]
    assert result.remaining_errors == []


@pytest.mark.asyncio
async def test_fix_with_remaining_errors(tool, session):
    """Fixer applies some fixes but validation still fails."""
    agent_json = {
        "nodes": [
            {
                "id": "node-1",
                "block_id": "block-1",
                "input_default": {},
                "metadata": {},
            }
        ],
        "links": [
            {
                "id": "link-1",
                "source_id": "node-1",
                "source_name": "output",
                "sink_id": "node-2",
                "sink_name": "input",
            }
        ],
    }

    fixed_agent = dict(agent_json)

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = AsyncMock(return_value=fixed_agent)
    mock_fixer.get_fixes_applied.return_value = ["Fixed UUID"]

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (
        False,
        "Link references non-existent node 'node-2'",
    )
    mock_validator.errors = ["Link references non-existent node 'node-2'"]

    with (
        patch(
            "backend.copilot.tools.fix_agent.get_blocks_as_dicts",
            return_value=[],
        ),
        patch(
            "backend.copilot.tools.fix_agent.AgentFixer",
            return_value=mock_fixer,
        ),
        patch(
            "backend.copilot.tools.fix_agent.AgentValidator",
            return_value=mock_validator,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, FixResultResponse)
    assert result.valid_after_fix is False
    assert result.fix_count == 1
    assert len(result.remaining_errors) == 1


@pytest.mark.asyncio
async def test_fixer_exception_returns_error(tool, session):
    """Fixer exception returns ErrorResponse."""
    agent_json = {
        "nodes": [{"id": "n1", "block_id": "b1", "input_default": {}, "metadata": {}}],
        "links": [],
    }

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = AsyncMock(side_effect=RuntimeError("fixer crashed"))

    with (
        patch(
            "backend.copilot.tools.fix_agent.get_blocks_as_dicts",
            return_value=[],
        ),
        patch(
            "backend.copilot.tools.fix_agent.AgentFixer",
            return_value=mock_fixer,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "fix_exception" in result.error
