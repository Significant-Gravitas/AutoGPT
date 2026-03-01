"""Tests for GetBlocksForGoalTool."""

from unittest.mock import patch

import pytest

from backend.copilot.tools.get_blocks import GetBlocksForGoalTool
from backend.copilot.tools.models import BlocksForGoalResponse, ErrorResponse

from ._test_data import make_session

_TEST_USER_ID = "test-user-get-blocks"


@pytest.fixture
def tool():
    return GetBlocksForGoalTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


@pytest.mark.asyncio
async def test_missing_goal_returns_error(tool, session):
    """Missing goal returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session, goal="")
    assert isinstance(result, ErrorResponse)
    assert result.error is not None
    assert "goal" in result.error.lower()


@pytest.mark.asyncio
async def test_valid_goal_returns_blocks(tool, session):
    """Valid goal returns BlocksForGoalResponse with blocks."""
    mock_blocks = [
        {
            "id": "block-1",
            "name": "AI Text Generator",
            "description": "Generates text using AI",
            "inputSchema": {"properties": {"prompt": {"type": "string"}}},
            "outputSchema": {"properties": {"response": {"type": "string"}}},
            "categories": [{"category": "AI"}],
            "staticOutput": False,
            "relevance_score": 3.0,
        },
        {
            "id": "block-2",
            "name": "Send Email",
            "description": "Sends an email",
            "inputSchema": {"properties": {"to": {"type": "string"}}},
            "outputSchema": {"properties": {"status": {"type": "string"}}},
            "categories": [{"category": "SOCIAL"}],
            "staticOutput": False,
            "relevance_score": 1.0,
        },
    ]

    with patch(
        "backend.copilot.tools.get_blocks.recommend_blocks_for_goal",
        return_value=mock_blocks,
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            goal="send email notifications with AI",
        )

    assert isinstance(result, BlocksForGoalResponse)
    assert result.count == 2
    assert result.goal == "send email notifications with AI"
    assert len(result.blocks) == 2
    assert result.blocks[0]["id"] == "block-1"
    assert result.blocks[1]["id"] == "block-2"


@pytest.mark.asyncio
async def test_max_blocks_parameter(tool, session):
    """max_blocks parameter is passed through to recommend_blocks_for_goal."""
    with patch(
        "backend.copilot.tools.get_blocks.recommend_blocks_for_goal",
        return_value=[],
    ) as mock_recommend:
        await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            goal="test goal",
            max_blocks=10,
        )

    mock_recommend.assert_called_once_with("test goal", max_blocks=10)


@pytest.mark.asyncio
async def test_recommendation_error_returns_error(tool, session):
    """Exception in recommend_blocks_for_goal returns ErrorResponse."""
    with patch(
        "backend.copilot.tools.get_blocks.recommend_blocks_for_goal",
        side_effect=RuntimeError("blocks not loaded"),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            goal="test goal",
        )

    assert isinstance(result, ErrorResponse)
    assert "Failed to load" in result.message
