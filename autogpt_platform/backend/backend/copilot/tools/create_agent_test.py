"""Tests for CreateAgentTool response types."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.tools.create_agent import CreateAgentTool
from backend.copilot.tools.models import (
    ClarificationNeededResponse,
    ErrorResponse,
    SuggestedGoalResponse,
)

from ._test_data import make_session

_TEST_USER_ID = "test-user-create-agent"


@pytest.fixture
def tool():
    return CreateAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


@pytest.mark.asyncio
async def test_missing_description_returns_error(tool, session):
    """Missing description returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session, description="")
    assert isinstance(result, ErrorResponse)
    assert result.error == "Missing description parameter"


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
