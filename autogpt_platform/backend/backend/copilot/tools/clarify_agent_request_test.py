"""Tests for ClarifyAgentRequestTool."""

import pytest

from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.clarify_agent_request import ClarifyAgentRequestTool
from backend.copilot.tools.models import (
    ClarificationNeededResponse,
    ErrorResponse,
    ResponseType,
)

_TEST_USER_ID = "test-user-clarify"


@pytest.fixture
def tool() -> ClarifyAgentRequestTool:
    return ClarifyAgentRequestTool()


@pytest.fixture
def session():
    return make_session(user_id=_TEST_USER_ID)


class TestClarifyAgentRequestTool:
    def test_name(self, tool: ClarifyAgentRequestTool):
        assert tool.name == "clarify_agent_request"

    def test_requires_no_auth(self, tool: ClarifyAgentRequestTool):
        assert tool.requires_auth is False

    def test_question_and_options_are_required(self, tool: ClarifyAgentRequestTool):
        params = tool.parameters
        assert "question" in params["required"]
        assert "options" in params["required"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_clarification_response(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="How should the agent deliver results?",
            options=["Email", "Slack"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.type == ResponseType.AGENT_BUILDER_CLARIFICATION_NEEDED
        assert response.message == "How should the agent deliver results?"
        assert response.session_id == session.session_id

    @pytest.mark.asyncio(loop_scope="session")
    async def test_question_is_set_on_clarifying_question(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What data source should the agent read from?",
            options=["RSS Feed", "Web Scraper"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert len(response.questions) == 1
        assert (
            response.questions[0].question
            == "What data source should the agent read from?"
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_options_joined_as_example(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="Where should results go?",
            options=["Email", "Slack", "Google Docs"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].example == "Email, Slack, Google Docs"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_single_option(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="Should the agent use Email?",
            options=["Email"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].example == "Email"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_options_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What trigger should start the agent?",
            options=[],
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_options"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_options_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What trigger should start the agent?",
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_options"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_blank_question_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="   ",
            options=["Email"],
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_question"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_question_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            options=["Email"],
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_question"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_none_session_uses_none_session_id(self, tool):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=None,
            question="How should the agent deliver results?",
            options=["Email", "Slack"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.session_id is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_keyword_defaults_to_empty_string(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What format should the output be?",
            options=["CSV", "JSON"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].keyword == ""
