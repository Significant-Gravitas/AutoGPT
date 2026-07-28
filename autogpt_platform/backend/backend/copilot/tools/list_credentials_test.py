"""Tests for the list_user_credentials tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.tools.list_credentials import (
    CredentialListResponse,
    ListUserCredentialsTool,
)
from backend.copilot.tools.models import ErrorResponse, ResponseType
from backend.data.model import APIKeyCredentials, OAuth2Credentials

# Shorthand patch target
_CREDS_PATH = "backend.copilot.tools.list_credentials.get_user_credentials"


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
    async def test_filters_sdk_default_credentials(self, tool, mock_session):
        creds = [_github_oauth(), _sdk_default_cred()]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert result.count == 1
        assert result.providers == ["github"]

    @pytest.mark.asyncio
    async def test_filters_system_credentials(self, tool, mock_session):
        from backend.integrations.credentials_store import DEFAULT_CREDENTIALS

        creds = [_notion_api_key(), *DEFAULT_CREDENTIALS]
        with patch(_CREDS_PATH, new_callable=AsyncMock, return_value=creds):
            result = await tool._execute(user_id="user-1", session=mock_session)

        assert result.count == 1
        assert result.providers == ["notion"]

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
