"""Tests for MCP API routes.

Uses httpx.AsyncClient with ASGITransport instead of fastapi.testclient.TestClient
to avoid creating blocking portals that can corrupt pytest-asyncio's session event loop.
"""

from unittest.mock import AsyncMock, patch

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id
from pydantic import SecretStr

from backend.api.features.mcp.routes import router
from backend.blocks.mcp.client import MCPClientError, MCPTool
from backend.data.model import OAuth2Credentials
from backend.util.request import HTTPClientError

app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: "test-user-id"


@pytest_asyncio.fixture(scope="module")
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _bypass_ssrf_validation():
    """Bypass validate_url in all route tests (test URLs don't resolve)."""
    with patch(
        "backend.api.features.mcp.routes.validate_url",
        new_callable=AsyncMock,
    ):
        yield


class TestDiscoverTools:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_success(self, client):
        mock_tools = [
            MCPTool(
                name="get_weather",
                description="Get weather for a city",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
            MCPTool(
                name="add_numbers",
                description="Add two numbers",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            ),
        ]

        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch(
                "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            instance = MockClient.return_value
            instance.initialize = AsyncMock(
                return_value={
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "test-server"},
                }
            )
            instance.list_tools = AsyncMock(return_value=mock_tools)

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://mcp.example.com/mcp"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["tools"]) == 2
        assert data["tools"][0]["name"] == "get_weather"
        assert data["tools"][1]["name"] == "add_numbers"
        assert data["server_name"] == "test-server"
        assert data["protocol_version"] == "2025-03-26"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_with_auth_token(self, client):
        with patch("backend.api.features.mcp.routes.MCPClient") as MockClient:
            instance = MockClient.return_value
            instance.initialize = AsyncMock(
                return_value={"serverInfo": {}, "protocolVersion": "2025-03-26"}
            )
            instance.list_tools = AsyncMock(return_value=[])

            response = await client.post(
                "/discover-tools",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "auth_token": "my-secret-token",
                },
            )

        assert response.status_code == 200
        MockClient.assert_called_once_with(
            "https://mcp.example.com/mcp",
            auth_token="my-secret-token",
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_auto_uses_stored_credential(self, client):
        """When no explicit token is given, stored MCP credentials are used."""
        stored_cred = OAuth2Credentials(
            provider="mcp",
            title="MCP: example.com",
            access_token=SecretStr("stored-token-123"),
            refresh_token=None,
            access_token_expires_at=None,
            refresh_token_expires_at=None,
            scopes=[],
            metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
        )

        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch(
                "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
                new_callable=AsyncMock,
                return_value=stored_cred,
            ),
        ):
            instance = MockClient.return_value
            instance.initialize = AsyncMock(
                return_value={"serverInfo": {}, "protocolVersion": "2025-03-26"}
            )
            instance.list_tools = AsyncMock(return_value=[])

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://mcp.example.com/mcp"},
            )

        assert response.status_code == 200
        MockClient.assert_called_once_with(
            "https://mcp.example.com/mcp",
            auth_token="stored-token-123",
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_mcp_error(self, client):
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch(
                "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            instance = MockClient.return_value
            instance.initialize = AsyncMock(
                side_effect=MCPClientError("Connection refused")
            )

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://bad-server.example.com/mcp"},
            )

        assert response.status_code == 502
        assert "Connection refused" in response.json()["detail"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_generic_error(self, client):
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch(
                "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            instance = MockClient.return_value
            instance.initialize = AsyncMock(side_effect=Exception("Network timeout"))

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://timeout.example.com/mcp"},
            )

        assert response.status_code == 502
        assert "Failed to connect" in response.json()["detail"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_auth_required(self, client):
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch(
                "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            instance = MockClient.return_value
            instance.initialize = AsyncMock(
                side_effect=HTTPClientError("HTTP 401 Error: Unauthorized", 401)
            )

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://auth-server.example.com/mcp"},
            )

        assert response.status_code == 401
        assert "requires authentication" in response.json()["detail"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_forbidden(self, client):
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch(
                "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            instance = MockClient.return_value
            instance.initialize = AsyncMock(
                side_effect=HTTPClientError("HTTP 403 Error: Forbidden", 403)
            )

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://auth-server.example.com/mcp"},
            )

        assert response.status_code == 401
        assert "requires authentication" in response.json()["detail"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_missing_url(self, client):
        response = await client.post("/discover-tools", json={})
        assert response.status_code == 422


class TestOAuthLogin:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_success(self, client):
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch(
                "backend.api.features.mcp.routes._register_mcp_client"
            ) as mock_register,
        ):
            instance = MockClient.return_value
            instance.discover_auth = AsyncMock(
                return_value={
                    "authorization_servers": ["https://auth.sentry.io"],
                    "resource": "https://mcp.sentry.dev/mcp",
                    "scopes_supported": ["openid"],
                }
            )
            instance.discover_auth_server_metadata = AsyncMock(
                return_value={
                    "authorization_endpoint": "https://auth.sentry.io/authorize",
                    "token_endpoint": "https://auth.sentry.io/token",
                    "registration_endpoint": "https://auth.sentry.io/register",
                }
            )
            mock_register.return_value = {
                "client_id": "registered-client-id",
                "client_secret": "registered-secret",
            }
            mock_cm.store.store_state_token = AsyncMock(
                return_value=("state-token-123", "code-challenge-abc")
            )
            mock_settings.config.frontend_base_url = "http://localhost:3000"

            response = await client.post(
                "/oauth/login",
                json={"server_url": "https://mcp.sentry.dev/mcp"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "login_url" in data
        assert data["state_token"] == "state-token-123"
        assert "auth.sentry.io/authorize" in data["login_url"]
        assert "registered-client-id" in data["login_url"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_no_oauth_support(self, client):
        with patch("backend.api.features.mcp.routes.MCPClient") as MockClient:
            instance = MockClient.return_value
            instance.discover_auth = AsyncMock(return_value=None)
            instance.discover_auth_server_metadata = AsyncMock(return_value=None)

            response = await client.post(
                "/oauth/login",
                json={"server_url": "https://simple-server.example.com/mcp"},
            )

        assert response.status_code == 400
        assert "does not advertise OAuth" in response.json()["detail"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_fallback_to_public_client(self, client):
        """When DCR is unavailable, falls back to default public client ID."""
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
        ):
            instance = MockClient.return_value
            instance.discover_auth = AsyncMock(
                return_value={
                    "authorization_servers": ["https://auth.example.com"],
                    "resource": "https://mcp.example.com/mcp",
                }
            )
            instance.discover_auth_server_metadata = AsyncMock(
                return_value={
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    # No registration_endpoint
                }
            )
            mock_cm.store.store_state_token = AsyncMock(
                return_value=("state-abc", "challenge-xyz")
            )
            mock_settings.config.frontend_base_url = "http://localhost:3000"

            response = await client.post(
                "/oauth/login",
                json={"server_url": "https://mcp.example.com/mcp"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "autogpt-platform" in data["login_url"]


class TestOAuthCallback:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_callback_success(self, client):
        mock_creds = OAuth2Credentials(
            provider="mcp",
            title=None,
            access_token=SecretStr("access-token-xyz"),
            refresh_token=None,
            access_token_expires_at=None,
            refresh_token_expires_at=None,
            scopes=[],
            metadata={
                "mcp_token_url": "https://auth.sentry.io/token",
                "mcp_resource_url": "https://mcp.sentry.dev/mcp",
            },
        )

        with (
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch("backend.api.features.mcp.routes.MCPOAuthHandler") as MockHandler,
        ):
            mock_settings.config.frontend_base_url = "http://localhost:3000"

            # Mock state verification
            mock_state = AsyncMock()
            mock_state.state_metadata = {
                "authorize_url": "https://auth.sentry.io/authorize",
                "token_url": "https://auth.sentry.io/token",
                "client_id": "test-client-id",
                "client_secret": "test-secret",
                "server_url": "https://mcp.sentry.dev/mcp",
            }
            mock_state.scopes = ["openid"]
            mock_state.code_verifier = "verifier-123"
            mock_cm.store.verify_state_token = AsyncMock(return_value=mock_state)
            mock_cm.create = AsyncMock()

            handler_instance = MockHandler.return_value
            handler_instance.exchange_code_for_tokens = AsyncMock(
                return_value=mock_creds
            )

            # Mock old credential cleanup
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])

            response = await client.post(
                "/oauth/callback",
                json={"code": "auth-code-abc", "state_token": "state-token-123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["provider"] == "mcp"
        assert data["type"] == "oauth2"
        mock_cm.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_callback_invalid_state(self, client):
        with patch("backend.api.features.mcp.routes.creds_manager") as mock_cm:
            mock_cm.store.verify_state_token = AsyncMock(return_value=None)

            response = await client.post(
                "/oauth/callback",
                json={"code": "auth-code", "state_token": "bad-state"},
            )

        assert response.status_code == 400
        assert "Invalid or expired" in response.json()["detail"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_callback_token_exchange_fails(self, client):
        with (
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch("backend.api.features.mcp.routes.MCPOAuthHandler") as MockHandler,
        ):
            mock_settings.config.frontend_base_url = "http://localhost:3000"
            mock_state = AsyncMock()
            mock_state.state_metadata = {
                "authorize_url": "https://auth.example.com/authorize",
                "token_url": "https://auth.example.com/token",
                "client_id": "cid",
                "server_url": "https://mcp.example.com/mcp",
            }
            mock_state.scopes = []
            mock_state.code_verifier = "v"
            mock_cm.store.verify_state_token = AsyncMock(return_value=mock_state)

            handler_instance = MockHandler.return_value
            handler_instance.exchange_code_for_tokens = AsyncMock(
                side_effect=RuntimeError("Token exchange failed")
            )

            response = await client.post(
                "/oauth/callback",
                json={"code": "bad-code", "state_token": "state"},
            )

        assert response.status_code == 400
        assert "token exchange failed" in response.json()["detail"].lower()


class TestStoreToken:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_success(self, client):
        with patch("backend.api.features.mcp.routes.creds_manager") as mock_cm:
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "my-api-key-123",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "mcp"
        assert data["type"] == "oauth2"
        assert data["host"] == "mcp.example.com"
        mock_cm.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_blank_rejected(self, client):
        """Blank token string (after stripping) should return 422."""
        response = await client.post(
            "/token",
            json={
                "server_url": "https://mcp.example.com/mcp",
                "token": "   ",
            },
        )
        # Pydantic min_length=1 catches the whitespace-only token
        assert response.status_code == 422

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_replaces_old_credential(self, client):
        old_cred = OAuth2Credentials(
            provider="mcp",
            title="MCP: mcp.example.com",
            access_token=SecretStr("old-token"),
            scopes=[],
            metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
        )
        with patch("backend.api.features.mcp.routes.creds_manager") as mock_cm:
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[old_cred])
            mock_cm.create = AsyncMock()
            mock_cm.store.delete_creds_by_id = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "new-token",
                },
            )

        assert response.status_code == 200
        mock_cm.store.delete_creds_by_id.assert_called_once_with(
            "test-user-id", old_cred.id
        )


class TestSSRFValidation:
    """Verify that validate_url is enforced on all endpoints."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_ssrf_blocked(self, client):
        with patch(
            "backend.api.features.mcp.routes.validate_url",
            new_callable=AsyncMock,
            side_effect=ValueError("blocked loopback"),
        ):
            response = await client.post(
                "/discover-tools",
                json={"server_url": "http://localhost/mcp"},
            )

        assert response.status_code == 400
        assert "blocked loopback" in response.json()["detail"].lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_ssrf_blocked(self, client):
        with patch(
            "backend.api.features.mcp.routes.validate_url",
            new_callable=AsyncMock,
            side_effect=ValueError("blocked private IP"),
        ):
            response = await client.post(
                "/oauth/login",
                json={"server_url": "http://10.0.0.1/mcp"},
            )

        assert response.status_code == 400
        assert "blocked private ip" in response.json()["detail"].lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_ssrf_blocked(self, client):
        with patch(
            "backend.api.features.mcp.routes.validate_url",
            new_callable=AsyncMock,
            side_effect=ValueError("blocked loopback"),
        ):
            response = await client.post(
                "/token",
                json={
                    "server_url": "http://127.0.0.1/mcp",
                    "token": "some-token",
                },
            )

        assert response.status_code == 400
        assert "blocked loopback" in response.json()["detail"].lower()
