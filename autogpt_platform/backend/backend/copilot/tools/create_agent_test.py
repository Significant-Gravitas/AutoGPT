"""Tests for CreateAgentTool response types."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.create_agent import CreateAgentTool
from backend.copilot.tools.models import (
    AgentPreviewResponse,
    ClarificationNeededResponse,
    ErrorResponse,
    SuggestedGoalResponse,
)

from ._test_data import make_session

_TEST_USER_ID = "test-user-create-agent"
_PIPELINE = "backend.copilot.tools.agent_generator.pipeline"


@pytest.fixture
def tool():
    return CreateAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── External mode tests (description only) ──────────────────────────────


@pytest.mark.asyncio
async def test_missing_description_and_json_returns_error(tool, session):
    """Missing both description and agent_json returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session, description="")
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_vague_goal_returns_suggested_goal_response(tool, session):
    """vague_goal decomposition result returns SuggestedGoalResponse, not ErrorResponse."""
    vague_result = {
        "type": "vague_goal",
        "suggested_goal": "Monitor Twitter mentions for a specific keyword and send a daily digest email",
    }

    with (
        patch(
            "backend.copilot.tools.create_agent.decompose_goal",
            new_callable=AsyncMock,
            return_value=vague_result,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            description="monitor social media",
        )

    assert isinstance(result, SuggestedGoalResponse)
    assert result.goal_type == "vague"
    assert result.suggested_goal == vague_result["suggested_goal"]
    assert result.original_goal == "monitor social media"
    assert result.reason == "The goal needs more specific details"
    assert not isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_unachievable_goal_returns_suggested_goal_response(tool, session):
    """unachievable_goal decomposition result returns SuggestedGoalResponse, not ErrorResponse."""
    unachievable_result = {
        "type": "unachievable_goal",
        "suggested_goal": "Summarize the latest news articles on a topic and send them by email",
        "reason": "There are no blocks for mind-reading.",
    }

    with (
        patch(
            "backend.copilot.tools.create_agent.decompose_goal",
            new_callable=AsyncMock,
            return_value=unachievable_result,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            description="read my mind",
        )

    assert isinstance(result, SuggestedGoalResponse)
    assert result.goal_type == "unachievable"
    assert result.suggested_goal == unachievable_result["suggested_goal"]
    assert result.original_goal == "read my mind"
    assert result.reason == unachievable_result["reason"]
    assert not isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_clarifying_questions_returns_clarification_needed_response(
    tool, session
):
    """clarifying_questions decomposition result returns ClarificationNeededResponse."""
    clarifying_result = {
        "type": "clarifying_questions",
        "questions": [
            {
                "question": "What platform should be monitored?",
                "keyword": "platform",
                "example": "Twitter, Reddit",
            }
        ],
    }

    with (
        patch(
            "backend.copilot.tools.create_agent.decompose_goal",
            new_callable=AsyncMock,
            return_value=clarifying_result,
        ),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            description="monitor social media and alert me",
        )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].keyword == "platform"


# ── Local mode tests (agent_json provided) ──────────────────────────────


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
            agent_json=agent_json,
            save=True,
        )

    assert isinstance(result, ErrorResponse)
    assert "logged in" in result.message.lower()
