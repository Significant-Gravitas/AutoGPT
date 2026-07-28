"""Tests for the list_user_credentials tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.tools.list_credentials import (
    CredentialListResponse,
    ListUserCredentialsTool,
    _ensure_managed_credentials_bounded,
)
from backend.copilot.tools.models import ErrorResponse, ResponseType
from backend.data.model import (
    APIKeyCredentials,
    HostScopedCredentials,
    OAuth2Credentials,
)
from backend.integrations.credentials_store import DEFAULT_CREDENTIALS

# Shorthand patch targets
_CREDS_PATH = "backend.copilot.tools.list_credentials.get_user_credentials"
_MANAGED_PATH = (
    "backend.copilot.tools.list_credentials._ensure_managed_credentials_bounded"
)


@pytest.fixture(autouse=True)
def stub_managed_credentials_sweep():
    with patch(_MANAGED_PATH, new_callable=AsyncMock, return_value=True) as mock_sweep:
        yield mock_sweep


def _github_oauth() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="11111111-1111-1111-1111-111111111111",
        provider="github",
        title="GitHub",
        username="octocat",
        access_token=SecretStr("gho_secret"),
        refresh_token=SecretStr("ghr_secret"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=["repo", "read:org"],
    )


def _notion_api_key() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="22222222-2222-2222-2222-222222222222",
        provider="notion",
        title="My Notion key",
        api_key=SecretStr("secret_notion"),
        expires_at=None,
    )


def _host_scoped() -> HostScopedCredentials:
    return HostScopedCredentials(
        id="33333333-3333-3333-3333-333333333333",
        provider="http",
        title="Internal API",
        host="api.example.com",
        headers={"Authorization": SecretStr("Bearer host_scoped_secret")},
    )


def _mcp_oauth() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="44444444-4444-4444-4444-444444444444",
        provider="mcp",
        title="Linear MCP",
        access_token=SecretStr("mcp_access_secret"),
        scopes=[],
        metadata={"mcp_server_url": "https://mcp.linear.app/mcp"},
    )


def _managed_cred() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="55555555-5555-5555-5555-555555555555",
        provider="agentmail",
        title="AgentMail (managed)",
        api_key=SecretStr("managed_secret"),
        expires_at=None,
        is_managed=True,
    )


def _sdk_default_cred() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="firecrawl-default",
        provider="firecrawl",
        title="Firecrawl (SDK default)",
        api_key=SecretStr("sdk_secret"),
        expires_at=None,
    )


@pytest.fixture
def tool():
    return ListUserCredentialsTool()


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.session_id = "test-session-123"
    return session


class TestListUserCredentialsTool:
    def test_name(self, tool):
        assert tool.name == "list_user_credentials"

    def test_requires_auth(self, tool):
        assert tool.requires_auth is True

    def test_is_available(self, tool):
        assert tool.is_available is True

    def test_parameters_schema(self, tool):
        params = tool.parameters
        assert params["type"] == "object"
        assert "provider" in params["properties"]
        assert params["required"] == []

    @pytest.mark.asyncio
    async def test_lists_connected_credentials(self, tool, mock_session):
        creds = [_github_oauth(), _notion_api_key()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert isinstance(result, CredentialListResponse)
        assert result.type == ResponseType.CREDENTIAL_LIST
        assert result.count == 2
        assert result.providers == ["github", "notion"]
        assert result.session_id == "test-session-123"
        assert "github" in result.message and "notion" in result.message

        github = next(c for c in result.credentials if c.provider == "github")
        assert github.type == "oauth2"
        assert github.username == "octocat"
        assert github.scopes == ["repo", "read:org"]

    @pytest.mark.asyncio
    async def test_no_secrets_in_serialized_output(self, tool, mock_session):
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=[_github_oauth()]):
            result = await tool._execute(user_id="user-1", session=mock_session)

        payload = result.model_dump_json()
        assert "gho_secret" not in payload
        assert "ghr_secret" not in payload

    @pytest.mark.asyncio
    async def test_host_scoped_and_mcp_credentials(self, tool, mock_session):
        creds = [_host_scoped(), _mcp_oauth()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert isinstance(result, CredentialListResponse)
        assert result.count == 2

        host_scoped = next(c for c in result.credentials if c.type == "host_scoped")
        assert host_scoped.host == "api.example.com"
        mcp = next(c for c in result.credentials if c.provider == "mcp")
        assert mcp.host == "https://mcp.linear.app/mcp"

        payload = result.model_dump_json()
        assert "host_scoped_secret" not in payload
        assert "Authorization" not in payload
        assert "mcp_access_secret" not in payload

    @pytest.mark.asyncio
    async def test_filters_sdk_default_credentials(self, tool, mock_session):
        creds = [_github_oauth(), _sdk_default_cred()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert result.count == 1
        assert result.providers == ["github"]

    @pytest.mark.asyncio
    async def test_filters_system_credentials(self, tool, mock_session):
        creds = [_notion_api_key(), *DEFAULT_CREDENTIALS]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert result.count == 1
        assert result.providers == ["notion"]

    @pytest.mark.asyncio
    async def test_managed_credentials_are_included_and_flagged(
        self, tool, mock_session
    ):
        creds = [_managed_cred()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert isinstance(result, CredentialListResponse)
        assert result.count == 1
        assert result.providers == ["agentmail"]
        assert result.credentials[0].is_managed is True

    @pytest.mark.asyncio
    async def test_skips_managed_sweep_for_non_managed_provider(
        self, tool, mock_session, stub_managed_credentials_sweep
    ):
        with patch(
            "backend.copilot.tools.list_credentials.get_managed_provider",
            return_value=None,
        ), patch(_CREDS_PATH, new_callable=AsyncMock, return_value=[_github_oauth()]):
            result = await tool._execute(
                user_id="user-1", session=mock_session, provider="github"
            )

        assert result.provisioning_complete is True
        stub_managed_credentials_sweep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provider_filter(self, tool, mock_session):
        creds = [_github_oauth(), _notion_api_key()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(
                user_id="user-1", session=mock_session, provider="GitHub "
            )

        assert result.count == 1
        assert result.providers == ["github"]

    @pytest.mark.asyncio
    async def test_provider_filter_no_match(self, tool, mock_session):
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=[_github_oauth()]):
            result = await tool._execute(
                user_id="user-1", session=mock_session, provider="google"
            )

        assert isinstance(result, CredentialListResponse)
        assert result.count == 0
        assert result.credentials == []
        assert "google" in result.message

    @pytest.mark.asyncio
    async def test_whitespace_provider_treated_as_absent(self, tool, mock_session):
        creds = [_github_oauth(), _notion_api_key()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(
                user_id="user-1", session=mock_session, provider="   "
            )

        assert isinstance(result, CredentialListResponse)
        assert result.count == 2
        assert result.providers == ["github", "notion"]

    @pytest.mark.asyncio
    async def test_provisioning_failure_marks_list_incomplete(
        self, tool, mock_session, stub_managed_credentials_sweep
    ):
        stub_managed_credentials_sweep.return_value = False
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=[_github_oauth()]):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert isinstance(result, CredentialListResponse)
        assert result.provisioning_complete is False
        assert "do not treat their absence as authoritative" in result.message
        stub_managed_credentials_sweep.assert_awaited_once_with("user-1")

    @pytest.mark.asyncio
    async def test_empty_credentials(self, tool, mock_session):
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=[]):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert isinstance(result, CredentialListResponse)
        assert result.count == 0
        assert "not connected any integrations" in result.message

    @pytest.mark.asyncio
    async def test_lookup_failure_returns_error(self, tool, mock_session):
        with patch(
            _CREDS_PATH, new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert isinstance(result, ErrorResponse)
        assert result.error == "credential_lookup_failed"

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_error(self, tool, mock_session):
        result = await tool._execute(user_id=None, session=mock_session)

        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_user_id"

    @pytest.mark.asyncio
    async def test_execute_without_auth_returns_need_login(self, tool, mock_session):
        result = await tool.execute(None, mock_session, "tool-call-1")

        assert result.toolName == "list_user_credentials"
        assert '"need_login"' in result.output


_MGR_PATH = "backend.copilot.tools.list_credentials.IntegrationCredentialsManager"
_ENSURE_PATH = "backend.copilot.tools.list_credentials.ensure_managed_credentials"


class TestEnsureManagedCredentialsBounded:
    """Exercises the real bounded provisioning sweep (not the autouse stub)."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        with patch(_MGR_PATH), patch(
            _ENSURE_PATH, new_callable=AsyncMock
        ) as mock_ensure:
            result = await _ensure_managed_credentials_bounded("user-1")

        assert result is True
        mock_ensure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(1)

        with patch(_MGR_PATH), patch(_ENSURE_PATH, _hang), patch(
            "backend.copilot.tools.list_credentials._MANAGED_PROVISION_TIMEOUT_S", 0.01
        ):
            result = await _ensure_managed_credentials_bounded("user-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self):
        with patch(_MGR_PATH), patch(
            _ENSURE_PATH, new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            result = await _ensure_managed_credentials_bounded("user-1")

        assert result is False
