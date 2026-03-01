"""Tests for CustomizeAgentTool local mode."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.customize_agent import CustomizeAgentTool
from backend.copilot.tools.models import AgentPreviewResponse, ErrorResponse

from ._test_data import make_session

_TEST_USER_ID = "test-user-customize-agent"
_PIPELINE = "backend.copilot.tools.agent_generator.pipeline"


@pytest.fixture
def tool():
    return CustomizeAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── Input validation tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_agent_id_returns_error(tool, session):
    """Missing agent_id returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session, agent_id="")
    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "missing_agent_id" in result.error


# ── Local mode tests (agent_json provided) ───────────────────────────────


@pytest.mark.asyncio
async def test_local_mode_empty_nodes_returns_error(tool, session):
    """Local mode with no nodes returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_id="creator/test-agent",
        agent_json={"nodes": [], "links": []},
    )
    assert isinstance(result, ErrorResponse)
    assert "no nodes" in result.message.lower()


@pytest.mark.asyncio
async def test_local_mode_preview(tool, session):
    """Local mode with save=false returns AgentPreviewResponse."""
    agent_json = {
        "name": "Customized Agent",
        "description": "A customized agent",
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

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = AsyncMock(return_value=agent_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    with (
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="creator/test-agent",
            agent_json=agent_json,
            save=False,
        )

    assert isinstance(result, AgentPreviewResponse)
    assert result.agent_name == "Customized Agent"
    assert result.node_count == 1


@pytest.mark.asyncio
async def test_local_mode_validation_failure(tool, session):
    """Local mode returns ErrorResponse when validation fails."""
    agent_json = {
        "nodes": [
            {
                "id": "node-1",
                "block_id": "bad-block",
                "input_default": {},
                "metadata": {},
            }
        ],
        "links": [],
    }

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = AsyncMock(return_value=agent_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (False, "Block 'bad-block' not found")
    mock_validator.errors = ["Block 'bad-block' not found"]

    with (
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="creator/test-agent",
            agent_json=agent_json,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "validation_failed"
    assert "Block 'bad-block' not found" in result.message


@pytest.mark.asyncio
async def test_local_mode_no_auth_returns_error(tool, session):
    """Local mode with save=true and no user returns ErrorResponse."""
    agent_json = {
        "nodes": [
            {
                "id": "node-1",
                "block_id": "block-1",
                "input_default": {},
                "metadata": {},
            }
        ],
        "links": [],
    }

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = AsyncMock(return_value=agent_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    with (
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
    ):
        result = await tool._execute(
            user_id=None,
            session=session,
            agent_id="creator/test-agent",
            agent_json=agent_json,
            save=True,
        )

    assert isinstance(result, ErrorResponse)
    assert "logged in" in result.message.lower()


# ── External mode tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_external_mode_missing_modifications_returns_error(tool, session):
    """External mode with missing modifications returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_id="creator/test-agent",
        modifications="",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "missing_modifications" in result.error


@pytest.mark.asyncio
async def test_external_mode_invalid_agent_id_format(tool, session):
    """External mode with bad agent ID format returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_id="invalid-format",
        modifications="make it faster",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "invalid_agent_id_format" in result.error
