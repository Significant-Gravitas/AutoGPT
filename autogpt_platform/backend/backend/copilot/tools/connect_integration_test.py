"""Tests for ConnectIntegrationTool."""

import pytest

from ._test_data import make_session
from .connect_integration import ConnectIntegrationTool
from .models import ErrorResponse, SetupRequirementsResponse

_TEST_USER_ID = "test-user-connect-integration"


class TestConnectIntegrationTool:
    def _make_tool(self) -> ConnectIntegrationTool:
        return ConnectIntegrationTool()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_provider_returns_error(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="nonexistent"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "unknown_provider"
        assert "nonexistent" in result.message
        assert "github" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_provider_returns_error(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider=""
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "unknown_provider"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_github_provider_returns_setup_response(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="github"
        )
        assert isinstance(result, SetupRequirementsResponse)
        assert result.setup_info.agent_name == "GitHub"
        assert result.setup_info.agent_id == "connect_github"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_github_has_missing_credentials_in_readiness(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="github"
        )
        assert isinstance(result, SetupRequirementsResponse)
        readiness = result.setup_info.user_readiness
        assert readiness.has_all_credentials is False
        assert readiness.ready_to_run is False
        assert "github_credentials" in readiness.missing_credentials

    @pytest.mark.asyncio(loop_scope="session")
    async def test_github_requirements_include_credential_entry(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="github"
        )
        assert isinstance(result, SetupRequirementsResponse)
        creds = result.setup_info.requirements["credentials"]
        assert len(creds) == 1
        assert creds[0]["provider"] == "github"
        assert creds[0]["id"] == "github_credentials"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_reason_appears_in_message(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        reason = "Needed to create a pull request."
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="github", reason=reason
        )
        assert isinstance(result, SetupRequirementsResponse)
        assert reason in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_session_id_propagated(self):
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="github"
        )
        assert isinstance(result, SetupRequirementsResponse)
        assert result.session_id == session.session_id

    @pytest.mark.asyncio(loop_scope="session")
    async def test_provider_case_insensitive(self):
        """Provider slug is normalised to lowercase before lookup."""
        tool = self._make_tool()
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool._execute(
            user_id=_TEST_USER_ID, session=session, provider="GitHub"
        )
        assert isinstance(result, SetupRequirementsResponse)

    def test_tool_name(self):
        assert ConnectIntegrationTool().name == "connect_integration"

    def test_requires_auth(self):
        assert ConnectIntegrationTool().requires_auth is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unauthenticated_user_gets_need_login_response(self):
        """execute() with user_id=None must return NeedLoginResponse, not the setup card.

        This verifies that the requires_auth guard in BaseTool.execute() fires
        before _execute() is called, so unauthenticated callers cannot probe
        which integrations are configured.
        """
        import json

        tool = self._make_tool()
        # Session still needs a user_id string; the None is passed to execute()
        # to simulate an unauthenticated call.
        session = make_session(user_id=_TEST_USER_ID)
        result = await tool.execute(
            user_id=None,
            session=session,
            tool_call_id="test-call-id",
            provider="github",
        )
        raw = result.output
        output = json.loads(raw) if isinstance(raw, str) else raw
        assert output.get("type") == "need_login"
        assert result.success is False
