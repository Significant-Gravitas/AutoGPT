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

    def test_all_three_params_are_required(self, tool: ClarifyAgentRequestTool):
        params = tool.parameters
        assert set(params["required"]) == {"dimension", "question", "options"}

    def test_dimension_has_enum_values(self, tool: ClarifyAgentRequestTool):
        dim_schema = tool.parameters["properties"]["dimension"]
        assert set(dim_schema["enum"]) == {
            "output_format",
            "delivery_channel",
            "data_source",
            "trigger",
        }

    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_clarification_response(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            dimension="delivery_channel",
            question="How should the agent deliver results?",
            options=["Email", "Slack"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.type == ResponseType.AGENT_BUILDER_CLARIFICATION_NEEDED
        assert response.message == "How should the agent deliver results?"
        assert response.session_id == session.session_id

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dimension_used_as_keyword(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            dimension="data_source",
            question="What data source should the agent read from?",
            options=["RSS Feed", "Web Scraper"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert len(response.questions) == 1
        assert response.questions[0].keyword == "data_source"
        assert (
            response.questions[0].question
            == "What data source should the agent read from?"
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_options_joined_as_example(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            dimension="delivery_channel",
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
            dimension="output_format",
            question="Should the agent use CSV format?",
            options=["CSV"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.questions[0].example == "CSV"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_options_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            dimension="trigger",
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
            dimension="trigger",
            question="What trigger should start the agent?",
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_options"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_blank_question_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            dimension="delivery_channel",
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
            dimension="delivery_channel",
            options=["Email"],
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_question"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_dimension_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            question="Where should results go?",
            options=["Email"],
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_dimension"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_blank_dimension_returns_error(self, tool, session):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            dimension="  ",
            question="Where should results go?",
            options=["Email"],
        )

        assert isinstance(response, ErrorResponse)
        assert response.error == "missing_dimension"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_none_session_uses_none_session_id(self, tool):
        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=None,
            dimension="delivery_channel",
            question="How should the agent deliver results?",
            options=["Email", "Slack"],
        )

        assert isinstance(response, ClarificationNeededResponse)
        assert response.session_id is None
