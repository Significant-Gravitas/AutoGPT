"""Tests for AskQuestionTool."""

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.ask_question import AskQuestionTool
from backend.copilot.tools.models import ClarificationNeededResponse


@pytest.fixture()
def tool() -> AskQuestionTool:
    return AskQuestionTool()


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession.new(user_id="test-user", dry_run=False)


@pytest.mark.asyncio
async def test_execute_with_options(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Which channel?",
        options=["Email", "Slack", "Google Docs"],
        keyword="channel",
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.message == "Which channel?"
    assert result.session_id == session.session_id
    assert len(result.questions) == 1

    q = result.questions[0]
    assert q.question == "Which channel?"
    assert q.keyword == "channel"
    assert q.example == "Email, Slack, Google Docs"


@pytest.mark.asyncio
async def test_execute_without_options(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        question="What format do you want?",
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.message == "What format do you want?"
    assert len(result.questions) == 1

    q = result.questions[0]
    assert q.question == "What format do you want?"
    assert q.keyword == ""
    assert q.example is None


@pytest.mark.asyncio
async def test_execute_with_keyword_only(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        question="How often should it run?",
        keyword="trigger",
    )

    assert isinstance(result, ClarificationNeededResponse)
    q = result.questions[0]
    assert q.keyword == "trigger"
    assert q.example is None
