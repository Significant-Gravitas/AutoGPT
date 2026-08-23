"""Tests for ConnectIntegrationTool."""

from unittest.mock import AsyncMock, patch

import pytest

from . import connect_integration as connect_integration_module
from ._test_data import make_session
from .connect_integration import ConnectIntegrationTool
from .models import ErrorResponse, SetupRequirementsResponse

_TEST_USER_ID = "test-user-connect-integration"


@pytest.fixture(autouse=True)
def stub_pending_connect():
    """The card write-through to Redis is asserted in its own tests below."""
    with patch.object(
        connect_integration_module, "record_pending_connect", new=AsyncMock()
    ) as stub:
        yield stub


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


class TestScopeMerge:
    """The card carries the scopes the OAuth flow will actually request, so
    what lands here is what the post-connect check diffs the grant against."""

    def _scopes(self, response: SetupRequirementsResponse) -> list[str]:
        return response.setup_info.requirements["credentials"][0]["scopes"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_defaults_used_when_the_model_asks_for_nothing(self):
        result = await ConnectIntegrationTool()._execute(
            user_id=_TEST_USER_ID,
            session=make_session(user_id=_TEST_USER_ID),
            provider="github",
        )
        assert isinstance(result, SetupRequirementsResponse)
        assert self._scopes(result) == ["repo"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_requested_scopes_are_merged_on_top_of_the_defaults(self):
        result = await ConnectIntegrationTool()._execute(
            user_id=_TEST_USER_ID,
            session=make_session(user_id=_TEST_USER_ID),
            provider="github",
            scopes=["workflow", "read:org"],
        )
        assert isinstance(result, SetupRequirementsResponse)
        assert set(self._scopes(result)) == {"repo", "workflow", "read:org"}

    @pytest.mark.asyncio(loop_scope="session")
    async def test_duplicate_and_blank_scopes_are_dropped(self, stub_pending_connect):
        result = await ConnectIntegrationTool()._execute(
            user_id=_TEST_USER_ID,
            session=make_session(user_id=_TEST_USER_ID),
            provider="github",
            scopes=["repo", "  ", "workflow", "workflow"],
        )
        assert isinstance(result, SetupRequirementsResponse)
        assert set(self._scopes(result)) == {"repo", "workflow"}
        # Order-preserving dedupe, defaults first.
        recorded = stub_pending_connect.await_args.kwargs["requested_scopes"]
        assert recorded == ["repo", "workflow"]


class TestPendingConnectRecord:
    """Rendering the card records what it asked for, so the OAuth callback
    can report a shortfall back into this chat."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_card_records_the_merged_scopes_against_the_session(
        self, stub_pending_connect
    ):
        session = make_session(user_id=_TEST_USER_ID)
        await ConnectIntegrationTool()._execute(
            user_id=_TEST_USER_ID,
            session=session,
            provider="github",
            scopes=["workflow"],
        )

        stub_pending_connect.assert_awaited_once()
        kwargs = stub_pending_connect.await_args.kwargs
        assert kwargs["user_id"] == _TEST_USER_ID
        assert kwargs["provider"] == "github"
        assert kwargs["session_id"] == session.session_id
        assert kwargs["requested_scopes"] == ["repo", "workflow"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_provider_slug_is_normalized_before_recording(
        self, stub_pending_connect
    ):
        await ConnectIntegrationTool()._execute(
            user_id=_TEST_USER_ID,
            session=make_session(user_id=_TEST_USER_ID),
            provider="GitHub",
        )
        assert stub_pending_connect.await_args.kwargs["provider"] == "github"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_provider_records_nothing(self, stub_pending_connect):
        await ConnectIntegrationTool()._execute(
            user_id=_TEST_USER_ID,
            session=make_session(user_id=_TEST_USER_ID),
            provider="nonexistent",
        )
        stub_pending_connect.assert_not_awaited()
