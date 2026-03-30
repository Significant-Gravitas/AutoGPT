"""Tests for ClarifyAgentRequestTool."""

import pytest

from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.clarify_agent_request import ClarifyAgentRequestTool
from backend.copilot.tools.models import ClarificationNeededResponse, ResponseType

_TEST_USER_ID = "test-user-clarify"


class TestClarifyAgentRequestTool:
    def test_name(self):
        assert ClarifyAgentRequestTool().name == "clarify_agent_request"

    def test_requires_no_auth(self):
        assert ClarifyAgentRequestTool().requires_auth is False

    def test_question_is_required_parameter(self):
        params = ClarifyAgentRequestTool().parameters
        assert "question" in params["required"]
        assert "options" not in params["required"]
        assert "keyword" not in params["required"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_clarification_response(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="How should the agent deliver results?",
            options=["Email", "Slack"],
            keyword="email send",
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.type == ResponseType.AGENT_BUILDER_CLARIFICATION_NEEDED
        assert response.message == "How should the agent deliver results?"
        assert response.session_id == session.session_id

    @pytest.mark.asyncio(loop_scope="session")
    async def test_question_is_set_on_clarifying_question(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What data source should the agent read from?",
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert len(response.questions) == 1
        assert (
            response.questions[0].question
            == "What data source should the agent read from?"
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_options_joined_as_example(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="Where should results go?",
            options=["Email", "Slack", "Google Docs"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].example == "Email, Slack, Google Docs"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_keyword_is_stored(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="How should the agent notify you?",
            keyword="slack message",
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].keyword == "slack message"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_options_gives_none_example(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What trigger should start the agent?",
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].example is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_options_list_gives_none_example(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What trigger should start the agent?",
            options=[],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].example is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_keyword_defaults_to_empty_string(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = ClarifyAgentRequestTool()

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="What format should the output be?",
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].keyword == ""
