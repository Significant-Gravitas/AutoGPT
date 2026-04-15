"""Tests for ValidateAgentGraphTool."""

from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.tools.models import ErrorResponse, ValidationResultResponse
from backend.copilot.tools.validate_agent import ValidateAgentGraphTool

from ._test_data import make_session

_TEST_USER_ID = "test-user-validate-agent"


@pytest.fixture
def tool():
    return ValidateAgentGraphTool()


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
async def test_valid_agent_returns_success(tool, session):
    """Valid agent returns ValidationResultResponse with valid=True."""
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

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    with (
        patch(
            "backend.copilot.tools.validate_agent.get_blocks_as_dicts",
            return_value=[],
        ),
        patch(
            "backend.copilot.tools.validate_agent.AgentValidator",
            return_value=mock_validator,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, ValidationResultResponse)
    assert result.valid is True
    assert result.error_count == 0
    assert result.errors == []


@pytest.mark.asyncio
async def test_invalid_agent_returns_errors(tool, session):
    """Invalid agent returns ValidationResultResponse with errors."""
    agent_json = {
        "nodes": [
            {
                "id": "node-1",
                "block_id": "nonexistent-block",
                "input_default": {},
                "metadata": {},
            }
        ],
        "links": [],
    }

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (False, "Validation failed")
    mock_validator.errors = [
        "Block 'nonexistent-block' not found in registry",
        "Missing required input field 'prompt'",
    ]

    with (
        patch(
            "backend.copilot.tools.validate_agent.get_blocks_as_dicts",
            return_value=[],
        ),
        patch(
            "backend.copilot.tools.validate_agent.AgentValidator",
            return_value=mock_validator,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, ValidationResultResponse)
    assert result.valid is False
    assert result.error_count == 2
    assert len(result.errors) == 2


@pytest.mark.asyncio
async def test_validation_exception_returns_error(tool, session):
    """Validator exception returns ErrorResponse."""
    agent_json = {
        "nodes": [{"id": "n1", "block_id": "b1", "input_default": {}, "metadata": {}}],
        "links": [],
    }

    mock_validator = MagicMock()
    mock_validator.validate.side_effect = RuntimeError("unexpected error")

    with (
        patch(
            "backend.copilot.tools.validate_agent.get_blocks_as_dicts",
            return_value=[],
        ),
        patch(
            "backend.copilot.tools.validate_agent.AgentValidator",
            return_value=mock_validator,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "validation_exception" in result.error
