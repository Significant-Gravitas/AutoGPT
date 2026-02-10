"""
Tests for MCP OAuth handler.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.mcp.client import MCPClient
from backend.blocks.mcp.oauth import MCPOAuthHandler
from backend.data.model import OAuth2Credentials


def _mock_response(json_data: dict, status: int = 200) -> MagicMock:
    """Create a mock Response with synchronous json() (matching Requests.Response)."""
    resp = MagicMock()
    resp.status = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = json_data
    return resp


class TestMCPOAuthHandler:
    """Tests for the MCPOAuthHandler."""

    def _make_handler(self, **overrides) -> MCPOAuthHandler:
        defaults = {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "redirect_uri": "https://app.example.com/callback",
            "authorize_url": "https://auth.example.com/authorize",
            "token_url": "https://auth.example.com/token",
        }
        defaults.update(overrides)
        return MCPOAuthHandler(**defaults)

    def test_get_login_url_basic(self):
        handler = self._make_handler()
        url = handler.get_login_url(
            scopes=["read", "write"],
            state="random-state-token",
            code_challenge="S256-challenge-value",
        )

        assert "https://auth.example.com/authorize?" in url
        assert "response_type=code" in url
        assert "client_id=test-client-id" in url
        assert "state=random-state-token" in url
        assert "code_challenge=S256-challenge-value" in url
        assert "code_challenge_method=S256" in url
        assert "scope=read+write" in url

    def test_get_login_url_with_resource(self):
        handler = self._make_handler(resource_url="https://mcp.example.com/mcp")
        url = handler.get_login_url(
            scopes=[], state="state", code_challenge="challenge"
        )

        assert "resource=https" in url

    def test_get_login_url_without_pkce(self):
        handler = self._make_handler()
        url = handler.get_login_url(scopes=["read"], state="state", code_challenge=None)

        assert "code_challenge" not in url
        assert "code_challenge_method" not in url

    @pytest.mark.asyncio(loop_scope="session")
    async def test_exchange_code_for_tokens(self):
        handler = self._make_handler()

        resp = _mock_response(
            {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        )

        with patch("backend.blocks.mcp.oauth.Requests") as MockRequests:
            instance = MockRequests.return_value
            instance.post = AsyncMock(return_value=resp)

            creds = await handler.exchange_code_for_tokens(
                code="auth-code",
                scopes=["read"],
                code_verifier="pkce-verifier",
            )

        assert isinstance(creds, OAuth2Credentials)
        assert creds.access_token.get_secret_value() == "new-access-token"
        assert creds.refresh_token is not None
        assert creds.refresh_token.get_secret_value() == "new-refresh-token"
        assert creds.scopes == ["read"]
        assert creds.access_token_expires_at is not None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_refresh_tokens(self):
        handler = self._make_handler()

        existing_creds = OAuth2Credentials(
            id="existing-id",
            provider="mcp",
            access_token=SecretStr("old-token"),
            refresh_token=SecretStr("old-refresh"),
            scopes=["read"],
            title="test",
        )

        resp = _mock_response(
            {
                "access_token": "refreshed-token",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }
        )

        with patch("backend.blocks.mcp.oauth.Requests") as MockRequests:
            instance = MockRequests.return_value
            instance.post = AsyncMock(return_value=resp)

            refreshed = await handler._refresh_tokens(existing_creds)

        assert refreshed.id == "existing-id"
        assert refreshed.access_token.get_secret_value() == "refreshed-token"
        assert refreshed.refresh_token is not None
        assert refreshed.refresh_token.get_secret_value() == "new-refresh"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_refresh_tokens_no_refresh_token(self):
        handler = self._make_handler()

        creds = OAuth2Credentials(
            provider="mcp",
            access_token=SecretStr("token"),
            scopes=["read"],
            title="test",
        )

        with pytest.raises(ValueError, match="No refresh token"):
            await handler._refresh_tokens(creds)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_revoke_tokens_no_url(self):
        handler = self._make_handler(revoke_url=None)

        creds = OAuth2Credentials(
            provider="mcp",
            access_token=SecretStr("token"),
            scopes=[],
            title="test",
        )

        result = await handler.revoke_tokens(creds)
        assert result is False

    @pytest.mark.asyncio(loop_scope="session")
    async def test_revoke_tokens_with_url(self):
        handler = self._make_handler(revoke_url="https://auth.example.com/revoke")

        creds = OAuth2Credentials(
            provider="mcp",
            access_token=SecretStr("token"),
            scopes=[],
            title="test",
        )

        resp = _mock_response({}, status=200)

        with patch("backend.blocks.mcp.oauth.Requests") as MockRequests:
            instance = MockRequests.return_value
            instance.post = AsyncMock(return_value=resp)

            result = await handler.revoke_tokens(creds)

        assert result is True


class TestMCPClientDiscovery:
    """Tests for MCPClient OAuth metadata discovery."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_auth_found(self):
        client = MCPClient("https://mcp.example.com/mcp")

        metadata = {
            "authorization_servers": ["https://auth.example.com"],
            "resource": "https://mcp.example.com/mcp",
        }

        resp = _mock_response(metadata, status=200)

        with patch("backend.blocks.mcp.client.Requests") as MockRequests:
            instance = MockRequests.return_value
            instance.get = AsyncMock(return_value=resp)

            result = await client.discover_auth()

        assert result is not None
        assert result["authorization_servers"] == ["https://auth.example.com"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_auth_not_found(self):
        client = MCPClient("https://mcp.example.com/mcp")

        resp = _mock_response({}, status=404)

        with patch("backend.blocks.mcp.client.Requests") as MockRequests:
            instance = MockRequests.return_value
            instance.get = AsyncMock(return_value=resp)

            result = await client.discover_auth()

        assert result is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_auth_server_metadata(self):
        client = MCPClient("https://mcp.example.com/mcp")

        server_metadata = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
            "code_challenge_methods_supported": ["S256"],
        }

        resp = _mock_response(server_metadata, status=200)

        with patch("backend.blocks.mcp.client.Requests") as MockRequests:
            instance = MockRequests.return_value
            instance.get = AsyncMock(return_value=resp)

            result = await client.discover_auth_server_metadata(
                "https://auth.example.com"
            )

        assert result is not None
        assert result["authorization_endpoint"] == "https://auth.example.com/authorize"
        assert result["token_endpoint"] == "https://auth.example.com/token"
