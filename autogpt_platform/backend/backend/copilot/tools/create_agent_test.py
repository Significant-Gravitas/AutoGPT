"""Tests for CreateAgentTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.create_agent import CreateAgentTool
from backend.copilot.tools.models import AgentPreviewResponse, ErrorResponse

from ._test_data import make_session

_TEST_USER_ID = "test-user-create-agent"
_PIPELINE = "backend.copilot.tools.agent_generator.pipeline"


@pytest.fixture
def tool():
    return CreateAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── Input validation tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_agent_json_returns_error(tool, session):
    """Missing agent_json returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_json"


# ── Local mode tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_mode_empty_nodes_returns_error(tool, session):
    """Local mode with no nodes returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_json={"nodes": [], "links": []},
    )
    assert isinstance(result, ErrorResponse)
    assert "no nodes" in result.message.lower()


@pytest.mark.asyncio
async def test_local_mode_preview(tool, session):
    """Local mode with save=false returns AgentPreviewResponse."""
    agent_json = {
        "name": "Test Agent",
        "description": "A test agent",
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
    mock_fixer.apply_all_fixes = MagicMock(return_value=agent_json)
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
            agent_json=agent_json,
            save=False,
        )

    assert isinstance(result, AgentPreviewResponse)
    assert result.agent_name == "Test Agent"
    assert result.node_count == 1


@pytest.mark.asyncio
async def test_local_mode_validation_failure(tool, session):
    """Local mode returns ErrorResponse when validation fails after fixing."""
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
    mock_fixer.apply_all_fixes = MagicMock(return_value=agent_json)
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
            agent_json=agent_json,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "validation_failed"
    assert result.details is not None
    assert "Block 'bad-block' not found" in result.details["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_hidden", [True, False])
async def test_is_hidden_passes_through_to_pipeline(tool, session, is_hidden):
    """The is_hidden param on create_agent must reach
    fix_validate_and_save unchanged so trigger agents land in the DB
    hidden. Regression for an earlier review finding that the kwarg
    plumbing was untested."""
    agent_json = {
        "name": "Trigger Watcher",
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

    mock_fix = MagicMock(name="fix_validate_and_save_result")
    with (
        patch(
            "backend.copilot.tools.create_agent.fetch_library_agents",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "backend.copilot.tools.create_agent.fix_validate_and_save",
            new=AsyncMock(return_value=mock_fix),
        ) as mock_fix_validate,
    ):
        await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
            is_hidden=is_hidden,
        )

    mock_fix_validate.assert_awaited_once()
    assert mock_fix_validate.call_args.kwargs["is_hidden"] is is_hidden


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
    mock_fixer.apply_all_fixes = MagicMock(return_value=agent_json)
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
            agent_json=agent_json,
            save=True,
        )

    assert isinstance(result, ErrorResponse)
    assert "logged in" in result.message.lower()
