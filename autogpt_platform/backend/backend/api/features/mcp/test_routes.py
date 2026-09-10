"""Tests for MCP API routes.

Uses httpx2.AsyncClient with ASGITransport instead of fastapi.testclient.TestClient
to avoid creating blocking portals that can corrupt pytest-asyncio's session event loop.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import httpx2
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id
from pydantic import SecretStr

from backend.api.features.mcp.routes import router
from backend.blocks.mcp.client import MCPClientError, MCPTool
from backend.data.model import OAuth2Credentials
from backend.util.request import HTTPClientError, HTTPServerError

app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: "test-user-id"


@pytest_asyncio.fixture(scope="module")
async def client():
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _bypass_ssrf_validation():
    """Bypass validate_url_host in all route tests (test URLs don't resolve)."""
    with patch(
        "backend.api.features.mcp.routes.validate_url_host",
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
            instance.close = AsyncMock()
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
        instance.close.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_with_auth_token(self, client):
        with patch("backend.api.features.mcp.routes.MCPClient") as MockClient:
            instance = MockClient.return_value
            instance.close = AsyncMock()
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
            authorization="Bearer my-secret-token",
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
            instance.close = AsyncMock()
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
            authorization="Bearer stored-token-123",
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
            instance.close = AsyncMock()
            instance.initialize = AsyncMock(
                side_effect=MCPClientError("Connection refused")
            )

            response = await client.post(
                "/discover-tools",
                json={"server_url": "https://bad-server.example.com/mcp"},
            )

        assert response.status_code == 502
        assert "Connection refused" in response.json()["detail"]
        # The session is released on the error path too, not just on success.
        instance.close.assert_awaited_once()

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
            instance.close = AsyncMock()
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
            instance.close = AsyncMock()
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
            instance.close = AsyncMock()
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
                return_value=(
                    {
                        "authorization_endpoint": "https://auth.sentry.io/authorize",
                        "token_endpoint": "https://auth.sentry.io/token",
                        "registration_endpoint": "https://auth.sentry.io/register",
                    },
                    "https://auth.sentry.io",
                )
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
                return_value=(
                    {
                        "authorization_endpoint": "https://auth.example.com/authorize",
                        "token_endpoint": "https://auth.example.com/token",
                        # No registration_endpoint
                    },
                    "https://auth.example.com",
                )
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

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_binds_issuer_and_iss_requirement(self, client):
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
        ):
            instance = MockClient.return_value
            instance.discover_auth = AsyncMock(
                return_value={
                    "authorization_servers": ["https://auth.example.com/"],
                    "resource": "https://mcp.example.com/mcp",
                }
            )
            instance.discover_auth_server_metadata = AsyncMock(
                return_value=(
                    {
                        "issuer": "https://auth.example.com",
                        "authorization_endpoint": "https://auth.example.com/authorize",
                        "token_endpoint": "https://auth.example.com/token",
                        "authorization_response_iss_parameter_supported": True,
                    },
                    "https://auth.example.com",
                )
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
        state_metadata = mock_cm.store.store_state_token.call_args.kwargs[
            "state_metadata"
        ]
        assert state_metadata["issuer"] == "https://auth.example.com"
        assert state_metadata["iss_required"] is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_rejects_issuer_naming_another_server(self, client):
        """RFC 8414 §3.3: a document naming another issuer is refused outright.

        Dropping just the issuer would leave ``iss_required`` false and make
        the callback's mismatch check a no-op, so a hostile authorization
        server could switch off RFC 9207 mix-up protection by claiming a
        well-known issuer.
        """
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
                return_value=(
                    {
                        "issuer": "https://evil.example.net",
                        "authorization_endpoint": "https://auth.example.com/authorize",
                        "token_endpoint": "https://auth.example.com/token",
                        "authorization_response_iss_parameter_supported": True,
                    },
                    "https://auth.example.com",
                )
            )
            mock_cm.store.store_state_token = AsyncMock(
                return_value=("state-abc", "challenge-xyz")
            )
            mock_settings.config.frontend_base_url = "http://localhost:3000"

            response = await client.post(
                "/oauth/login",
                json={"server_url": "https://mcp.example.com/mcp"},
            )

        assert response.status_code == 400
        assert "does not match where the metadata was published" in (
            response.json()["detail"]
        )
        mock_cm.store.store_state_token.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_accepts_issuer_differing_only_in_case(self, client):
        """Scheme and host are case-insensitive (RFC 3986), so this is a match."""
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
                return_value=(
                    {
                        "issuer": "https://AUTH.Example.COM",
                        "authorization_endpoint": "https://auth.example.com/authorize",
                        "token_endpoint": "https://auth.example.com/token",
                    },
                    "https://auth.example.com",
                )
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
        state_metadata = mock_cm.store.store_state_token.call_args.kwargs[
            "state_metadata"
        ]
        assert state_metadata["issuer"] == "https://AUTH.Example.COM"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_rejects_tenant_issuer_naming_bare_origin(self, client):
        """A tenant's document may not claim the whole origin as its issuer.

        The expected issuer comes from whichever well-known URL answered, so a
        document fetched for ``/tenantA`` cannot pass by naming the origin
        that ``/tenantB`` also lives under.
        """
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
        ):
            instance = MockClient.return_value
            instance.discover_auth = AsyncMock(
                return_value={
                    "authorization_servers": ["https://auth.example.com/tenantA"],
                    "resource": "https://mcp.example.com/mcp",
                }
            )
            instance.discover_auth_server_metadata = AsyncMock(
                return_value=(
                    {
                        "issuer": "https://auth.example.com",
                        "authorization_endpoint": "https://auth.example.com/authorize",
                        "token_endpoint": "https://auth.example.com/token",
                    },
                    "https://auth.example.com/tenantA",
                )
            )
            mock_cm.store.store_state_token = AsyncMock(
                return_value=("state-abc", "challenge-xyz")
            )
            mock_settings.config.frontend_base_url = "http://localhost:3000"

            response = await client.post(
                "/oauth/login",
                json={"server_url": "https://mcp.example.com/mcp"},
            )

        assert response.status_code == 400
        mock_cm.store.store_state_token.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_allows_metadata_without_an_issuer(self, client):
        """No issuer claimed means no mix-up protection, not a hostile server."""
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
                return_value=(
                    {
                        "authorization_endpoint": "https://auth.example.com/authorize",
                        "token_endpoint": "https://auth.example.com/token",
                    },
                    "https://auth.example.com",
                )
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
        state_metadata = mock_cm.store.store_state_token.call_args.kwargs[
            "state_metadata"
        ]
        assert state_metadata["issuer"] == ""
        assert state_metadata["iss_required"] is False

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth_login_ignores_resource_on_another_origin(self, client):
        """RFC 9728 §3.3: a hostile server must not pick the token audience."""
        with (
            patch("backend.api.features.mcp.routes.MCPClient") as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
        ):
            instance = MockClient.return_value
            instance.discover_auth = AsyncMock(
                return_value={
                    "authorization_servers": ["https://as.legit.example"],
                    "resource": "https://api.legit.example/mcp",
                }
            )
            instance.discover_auth_server_metadata = AsyncMock(
                return_value=(
                    {
                        "authorization_endpoint": "https://as.legit.example/authorize",
                        "token_endpoint": "https://as.legit.example/token",
                    },
                    "https://as.legit.example",
                )
            )
            mock_cm.store.store_state_token = AsyncMock(
                return_value=("state-abc", "challenge-xyz")
            )
            mock_settings.config.frontend_base_url = "http://localhost:3000"

            response = await client.post(
                "/oauth/login",
                json={"server_url": "https://evil.example/mcp"},
            )

        assert response.status_code == 200
        state_metadata = mock_cm.store.store_state_token.call_args.kwargs[
            "state_metadata"
        ]
        assert state_metadata["resource_url"] == "https://evil.example/mcp"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dynamic_client_registration_declares_web_application(self):
        from backend.api.features.mcp.routes import _register_mcp_client

        with patch("backend.api.features.mcp.routes.Requests") as MockRequests:
            post = MockRequests.return_value.post = AsyncMock()
            post.return_value = MagicMock()
            post.return_value.json.return_value = {"client_id": "cid"}

            result = await _register_mcp_client(
                "https://auth.example.com/register",
                "http://localhost:3000/auth/integrations/mcp_callback",
                "https://mcp.example.com/mcp",
            )

        assert result == {"client_id": "cid"}
        payload = post.call_args.kwargs["json"]
        assert payload["application_type"] == "web"
        assert payload["redirect_uris"] == [
            "http://localhost:3000/auth/integrations/mcp_callback"
        ]


def _callback_mocks(mock_cm, mock_settings, MockHandler, state_metadata):
    """Wire the mocks a successful ``/oauth/callback`` needs."""
    mock_settings.config.frontend_base_url = "http://localhost:3000"
    mock_state = AsyncMock()
    mock_state.state_metadata = {
        "authorize_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "client_id": "cid",
        "server_url": "https://mcp.example.com/mcp",
        **state_metadata,
    }
    mock_state.scopes = []
    mock_state.code_verifier = "v"
    mock_cm.store.verify_state_token = AsyncMock(return_value=mock_state)
    mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
    mock_cm.create = AsyncMock()
    creds = OAuth2Credentials(
        provider="mcp",
        title=None,
        access_token=SecretStr("access-token"),
        refresh_token=None,
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=[],
        metadata={},
    )
    MockHandler.return_value.exchange_code_for_tokens = AsyncMock(return_value=creds)
    return creds


class TestOAuthCallbackIssuer:
    """RFC 9207 ``iss`` handling on the callback."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_mismatched_iss_is_rejected(self, client):
        with (
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch("backend.api.features.mcp.routes.MCPOAuthHandler") as MockHandler,
        ):
            _callback_mocks(
                mock_cm,
                mock_settings,
                MockHandler,
                {"issuer": "https://auth.example.com", "iss_required": True},
            )
            response = await client.post(
                "/oauth/callback",
                json={
                    "code": "code",
                    "state_token": "state",
                    "iss": "https://evil.example.net",
                },
            )

        assert response.status_code == 400
        assert "issuer does not match" in response.json()["detail"]
        MockHandler.return_value.exchange_code_for_tokens.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_iss_is_rejected_when_server_advertises_it(self, client):
        with (
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch("backend.api.features.mcp.routes.MCPOAuthHandler") as MockHandler,
        ):
            _callback_mocks(
                mock_cm,
                mock_settings,
                MockHandler,
                {"issuer": "https://auth.example.com", "iss_required": True},
            )
            response = await client.post(
                "/oauth/callback",
                json={"code": "code", "state_token": "state"},
            )

        assert response.status_code == 400
        assert "missing the issuer" in response.json()["detail"]
        MockHandler.return_value.exchange_code_for_tokens.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_iss_is_accepted_for_servers_that_do_not_send_it(
        self, client
    ):
        with (
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch("backend.api.features.mcp.routes.MCPOAuthHandler") as MockHandler,
        ):
            creds = _callback_mocks(
                mock_cm,
                mock_settings,
                MockHandler,
                {"issuer": "https://auth.example.com", "iss_required": False},
            )
            response = await client.post(
                "/oauth/callback",
                json={"code": "code", "state_token": "state"},
            )

        assert response.status_code == 200
        assert creds.metadata is not None
        assert creds.metadata["mcp_issuer"] == "https://auth.example.com"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_matching_iss_binds_issuer_to_credential(self, client):
        with (
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
            patch("backend.api.features.mcp.routes.settings") as mock_settings,
            patch("backend.api.features.mcp.routes.MCPOAuthHandler") as MockHandler,
        ):
            creds = _callback_mocks(
                mock_cm,
                mock_settings,
                MockHandler,
                {"issuer": "https://auth.example.com", "iss_required": True},
            )
            response = await client.post(
                "/oauth/callback",
                json={
                    "code": "code",
                    "state_token": "state",
                    "iss": "https://auth.example.com",
                },
            )

        assert response.status_code == 200
        mock_cm.create.assert_awaited_once()
        assert creds.metadata is not None
        assert creds.metadata["mcp_issuer"] == "https://auth.example.com"
        assert creds.metadata["mcp_server_url"] == "https://mcp.example.com/mcp"


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


def _probe_client(error: Exception | None = None):
    client = AsyncMock()
    client.initialize = AsyncMock(side_effect=error)
    client.close = AsyncMock()
    return client


def _probe_server(error: Exception | None = None):
    """Patch the ``/token`` verification probe. No *error* means the server
    accepts the token."""
    return patch(
        "backend.api.features.mcp.routes.MCPClient",
        return_value=_probe_client(error),
    )


def _accepting_server():
    return _probe_server()


class TestStoreToken:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_success(self, client):
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
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
        # ``host`` carries the full normalized ``mcp_server_url`` (not just
        # the bare hostname) so the response is parity with the OAuth
        # callback path — ``MCPSetupCard`` matches against this URL to
        # render the Connected/Reconnect state on chat refresh.
        assert data["host"] == "https://mcp.example.com/mcp"
        mock_cm.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_stored_manual_token_is_reused_by_discovery(self, client):
        """A manual token must survive the real auto-lookup path on retry."""
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            store_response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "Basic encoded-value",
                },
            )

        assert store_response.status_code == 200
        create_call = mock_cm.create.await_args
        assert create_call is not None
        stored_credential = create_call.args[1]
        assert isinstance(stored_credential, OAuth2Credentials)
        assert stored_credential.access_token_expires_at is None

        with (
            patch(
                "backend.blocks.mcp.helpers.IntegrationCredentialsManager"
            ) as manager_cls,
            patch("backend.api.features.mcp.routes.MCPClient") as client_cls,
        ):
            manager = manager_cls.return_value
            manager.store.get_creds_by_provider = AsyncMock(
                return_value=[stored_credential]
            )
            manager.refresh_if_needed = AsyncMock(
                side_effect=AssertionError("manual credentials must not refresh")
            )
            mcp_client = client_cls.return_value
            mcp_client.initialize = AsyncMock(
                return_value={
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "test-server"},
                }
            )
            mcp_client.list_tools = AsyncMock(return_value=[])
            mcp_client.close = AsyncMock()

            discover_response = await client.post(
                "/discover-tools",
                json={"server_url": "https://mcp.example.com/mcp"},
            )

        assert discover_response.status_code == 200
        client_cls.assert_called_once_with(
            "https://mcp.example.com/mcp", authorization="Basic encoded-value"
        )
        mcp_client.close.assert_awaited_once()
        manager.refresh_if_needed.assert_not_awaited()

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
    async def test_store_token_updates_existing_credential_in_place(self, client):
        old_cred = OAuth2Credentials(
            provider="mcp",
            title="MCP: mcp.example.com",
            access_token=SecretStr("old-token"),
            scopes=[],
            metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
        )
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[old_cred])
            mock_cm.create = AsyncMock()
            mock_cm.update = AsyncMock()
            mock_cm.delete = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "new-token",
                },
            )

        assert response.status_code == 200
        assert response.json()["id"] == old_cred.id
        mock_cm.create.assert_not_awaited()
        mock_cm.update.assert_awaited_once()
        update_call = mock_cm.update.await_args
        assert update_call is not None
        user_id, updated = update_call.args
        assert user_id == "test-user-id"
        assert updated.id == old_cred.id
        assert updated.access_token.get_secret_value() == "Bearer new-token"
        assert updated.metadata["mcp_auth_scheme"] == "bearer"
        mock_cm.delete.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_rotates_one_row_and_prunes_duplicates(self, client):
        """Rotation touches exactly one row and lets the duplicates go.

        Every ``update`` is a read-modify-write of the user's whole credential
        set, so updating each duplicate in turn multiplied the writes by the
        number of stale rows and left the row count growing forever.
        """
        old_creds = [
            OAuth2Credentials(
                provider="mcp",
                title=f"MCP credential {index}",
                access_token=SecretStr(f"old-token-{index}"),
                scopes=["existing-scope"],
                metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
            )
            for index in range(2)
        ]
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=old_creds)
            mock_cm.create = AsyncMock()
            mock_cm.update = AsyncMock()
            mock_cm.delete = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "Basic encoded-value",
                },
            )

        # `get_creds_by_provider` promises no ordering, so the survivor is
        # picked by an explicit key rather than by position in the response.
        survivor, superseded = sorted(old_creds, key=lambda cred: cred.id)[::-1]

        assert response.status_code == 200
        assert response.json()["id"] == survivor.id
        assert response.json()["mcp_auth_scheme"] == "basic"
        mock_cm.create.assert_not_awaited()
        mock_cm.update.assert_awaited_once()
        assert mock_cm.update.await_args is not None
        updated = mock_cm.update.await_args.args[1]
        assert updated.id == survivor.id
        assert updated.access_token.get_secret_value() == "Basic encoded-value"
        # A static pasted credential carries no OAuth scopes.
        assert updated.scopes == []
        mock_cm.delete.assert_awaited_once_with("test-user-id", superseded.id)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_replaces_an_oauth_credential_instead_of_converting_it(
        self, client
    ):
        """An OAuth row is neither rewritten in place nor deleted.

        Rewriting one would keep its id and ``type="oauth2"`` while discarding
        the refresh token and the client registration behind it, so a saved
        graph would silently start running on a pasted static secret.

        Deleting it is no better: neither ``creds_manager.delete`` nor
        ``delete_acquired`` calls ``handler.revoke_tokens`` -- that lives in
        ``DELETE /credentials`` -- so the row would go while a live refresh
        token stayed at the provider with nothing left to revoke it with, and
        every saved graph node bound to its id would break.  The manual
        credential wins in ``auto_lookup_mcp_credential`` instead.
        """
        oauth_cred = OAuth2Credentials(
            provider="mcp",
            title="MCP: mcp.example.com",
            access_token=SecretStr("oauth-access"),
            refresh_token=SecretStr("oauth-refresh"),
            scopes=["read"],
            metadata={
                "mcp_server_url": "https://mcp.example.com/mcp",
                "mcp_token_url": "https://mcp.example.com/token",
                "mcp_client_id": "client-abc",
            },
        )
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[oauth_cred])
            mock_cm.create = AsyncMock()
            mock_cm.update = AsyncMock()
            mock_cm.delete = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "pasted-token",
                },
            )

        assert response.status_code == 200
        mock_cm.update.assert_not_awaited()
        mock_cm.create.assert_awaited_once()
        assert mock_cm.create.await_args is not None
        created = mock_cm.create.await_args.args[1]
        assert created.id != oauth_cred.id
        assert created.refresh_token is None
        assert created.scopes == []
        mock_cm.delete.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_clears_stale_expiry_on_rotation(self, client):
        """Positive guard: a rotated manual credential must not stay expiring.

        Deleting the clearing left every assertion green, because only the
        negative "refresh is not attempted" case was covered.
        """
        old_cred = OAuth2Credentials(
            provider="mcp",
            title="MCP: mcp.example.com",
            access_token=SecretStr("old-token"),
            access_token_expires_at=1,
            username="someone",
            scopes=[],
            metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
        )
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[old_cred])
            mock_cm.create = AsyncMock()
            mock_cm.update = AsyncMock()
            mock_cm.delete = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "new-token",
                },
            )

        assert response.status_code == 200
        assert mock_cm.update.await_args is not None
        updated = mock_cm.update.await_args.args[1]
        assert updated.access_token_expires_at is None
        assert updated.username is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_does_not_mutate_managed_matching_credential(
        self, client
    ):
        managed = OAuth2Credentials(
            provider="mcp",
            title="Managed MCP",
            access_token=SecretStr("managed-token"),
            scopes=[],
            metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
            is_managed=True,
        )
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[managed])
            mock_cm.create = AsyncMock()
            mock_cm.update = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "user-token",
                },
            )

        assert response.status_code == 200
        mock_cm.update.assert_not_awaited()
        mock_cm.create.assert_awaited_once()
        create_call = mock_cm.create.await_args
        assert create_call is not None
        created = create_call.args[1]
        assert created.id != managed.id

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_fails_closed_when_existing_lookup_fails(self, client):
        with (
            _accepting_server(),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(
                side_effect=RuntimeError("database unavailable")
            )
            mock_cm.create = AsyncMock()
            mock_cm.update = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "new-token",
                },
            )

        assert response.status_code == 503
        mock_cm.create.assert_not_awaited()
        mock_cm.update.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_rejected_by_server_is_not_stored(self, client):
        """A 2xx here turns the setup card green, so it must mean the token
        actually authenticates — not merely that a row was written
        (SECRT-2592)."""
        with (
            _probe_server(HTTPClientError("HTTP 401", status_code=401)),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "wrong-token",
                },
            )

        assert response.status_code == 400
        assert "rejected this credential" in response.json()["detail"]
        mock_cm.create.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_survives_server_outage(self, client):
        """A 5xx says nothing about the token — refusing the save would strand
        the user during someone else's outage."""
        with (
            _probe_server(HTTPServerError("HTTP 503", status_code=503)),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "probably-fine",
                },
            )

        assert response.status_code == 200
        mock_cm.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_verifies_with_the_submitted_token(self, client):
        """The probe must carry the token — verifying anonymously would prove
        nothing."""
        probe = _probe_client()
        with (
            patch(
                "backend.api.features.mcp.routes.MCPClient", return_value=probe
            ) as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "my-api-key-123",
                },
            )

        MockClient.assert_called_once_with(
            "https://mcp.example.com/mcp",
            authorization="Bearer my-api-key-123",
            follow_redirects=False,
        )
        probe.close.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize("status_code", [403, 404, 429])
    async def test_store_token_stores_when_status_is_not_a_rejection(
        self, client, status_code
    ):
        """Only a 401 unambiguously means "this credential was refused".

        A bare 403 is as often a per-scope decision or a WAF blocking our
        egress IP; a 404 is a wrong MCP path. Blocking the save on those would
        tell the user their token is wrong when it isn't.
        """
        with (
            _probe_server(
                HTTPClientError(f"HTTP {status_code}", status_code=status_code)
            ),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "probably-fine",
                },
            )

        assert response.status_code == 200
        mock_cm.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_rejects_explicit_http(self, client):
        """The credential rides in an Authorization header — never cleartext."""
        with patch("backend.api.features.mcp.routes.creds_manager") as mock_cm:
            mock_cm.create = AsyncMock()
            response = await client.post(
                "/token",
                json={
                    "server_url": "http://mcp.example.com/mcp",
                    "token": "my-api-key-123",
                },
            )

        assert response.status_code == 400
        assert "https" in response.json()["detail"].lower()
        mock_cm.create.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_defaults_a_scheme_less_url_to_https(self, client):
        """Typing no scheme is not asking for cleartext, so it must not be
        rejected as though the user had written http://."""
        probe = _probe_client()
        with (
            patch(
                "backend.api.features.mcp.routes.MCPClient", return_value=probe
            ) as MockClient,
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "mcp.example.com/mcp",
                    "token": "my-api-key-123",
                },
            )

        assert response.status_code == 200
        assert response.json()["host"] == "https://mcp.example.com/mcp"
        assert MockClient.call_args.args[0] == "https://mcp.example.com/mcp"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_token_survives_a_hanging_server(self, client):
        """MCPClient sets no timeout and no retry ceiling, so an unbounded
        probe could pin the handler forever."""
        hanging = _probe_client()
        cancelled = asyncio.Event()

        async def never_returns():
            # Raising TimeoutError directly would pass with ``wait_for``
            # removed; the probe has to actually hang for this to mean
            # anything.
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        hanging.initialize = never_returns
        with (
            patch("backend.api.features.mcp.routes.MCPClient", return_value=hanging),
            patch("backend.api.features.mcp.routes._PROBE_TIMEOUT_SECONDS", 0.01),
            patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
        ):
            mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_cm.create = AsyncMock()

            response = await client.post(
                "/token",
                json={
                    "server_url": "https://mcp.example.com/mcp",
                    "token": "probably-fine",
                },
            )

        assert response.status_code == 200
        mock_cm.create.assert_called_once()
        assert cancelled.is_set()


class TestSSRFValidation:
    """Verify that validate_url_host is enforced on all endpoints."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_discover_tools_ssrf_blocked(self, client):
        with patch(
            "backend.api.features.mcp.routes.validate_url_host",
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
            "backend.api.features.mcp.routes.validate_url_host",
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
            "backend.api.features.mcp.routes.validate_url_host",
            new_callable=AsyncMock,
            side_effect=ValueError("blocked loopback"),
        ):
            response = await client.post(
                "/token",
                json={
                    "server_url": "https://127.0.0.1/mcp",
                    "token": "some-token",
                },
            )

        assert response.status_code == 400
        assert "blocked loopback" in response.json()["detail"].lower()
