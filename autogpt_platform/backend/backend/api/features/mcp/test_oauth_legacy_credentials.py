from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.mcp.oauth import MCPOAuthHandler
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import create_mcp_oauth_handler
from backend.util.request import HTTPClientError


@pytest.mark.parametrize("client_secret", ["", "<legacy-test-secret>"])
@pytest.mark.asyncio(loop_scope="session")
async def test_existing_credentials_without_auth_method_still_refresh(client_secret):
    credentials = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("<test-access-token>"),
        refresh_token=SecretStr("<test-refresh-token>"),
        scopes=["read"],
        metadata={
            "mcp_token_url": "https://auth.example.com/token",
            "mcp_client_id": "legacy-client",
            "mcp_client_secret": client_secret,
        },
    )
    with patch("backend.blocks.mcp.oauth.Requests") as requests:
        response = MagicMock()
        response.json.return_value = {"access_token": "<new-test-token>"}
        post = requests.return_value.post = AsyncMock(return_value=response)
        refreshed = await create_mcp_oauth_handler(credentials)._refresh_tokens(
            credentials
        )

    assert "Authorization" not in post.call_args.kwargs["headers"]
    data = post.call_args.kwargs["data"]
    assert data["client_id"] == "legacy-client"
    if client_secret:
        assert data["client_secret"] == client_secret
    else:
        assert "client_secret" not in data
    assert refreshed.id == credentials.id
    assert refreshed.refresh_token == credentials.refresh_token
    assert refreshed.metadata == credentials.metadata


@pytest.mark.parametrize(
    "method", ["client_secret_basic", "client_secret_post", "none"]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_revocation_uses_registered_authentication(method):
    handler = MCPOAuthHandler(
        client_id="client",
        client_secret="<test-secret>",
        redirect_uri="",
        authorize_url="",
        token_url="https://auth.example.com/token",
        revoke_url="https://auth.example.com/revoke",
        token_endpoint_auth_method=method,
    )
    credentials = OAuth2Credentials(
        provider="mcp", access_token=SecretStr("<test-token>"), scopes=[]
    )
    with patch("backend.blocks.mcp.oauth.Requests") as requests:
        post = requests.return_value.post = AsyncMock()
        assert await handler.revoke_tokens(credentials)

    args = post.call_args.kwargs
    assert args["data"]["token"] == "<test-token>"
    if method == "client_secret_basic":
        assert args["headers"]["Authorization"].startswith("Basic ")
        assert "client_id" not in args["data"]
        assert "client_secret" not in args["data"]
    elif method == "client_secret_post":
        assert "Authorization" not in args["headers"]
        assert args["data"]["client_id"] == "client"
        assert args["data"]["client_secret"] == "<test-secret>"
    else:
        assert "Authorization" not in args["headers"]
        assert args["data"]["client_id"] == "client"
        assert "client_secret" not in args["data"]


def test_unsupported_stored_authentication_method_is_rejected():
    credentials = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("<test-token>"),
        scopes=[],
        metadata={
            "mcp_token_url": "https://auth.example.com/token",
            "mcp_client_id": "client",
            "mcp_token_endpoint_auth_method": "private_key_jwt",
        },
    )
    with pytest.raises(
        ValueError, match="Unsupported MCP client authentication method"
    ):
        create_mcp_oauth_handler(credentials)


@pytest.mark.asyncio(loop_scope="session")
async def test_rejected_revocation_does_not_report_success():
    handler = MCPOAuthHandler(
        client_id="client",
        client_secret="",
        redirect_uri="",
        authorize_url="",
        token_url="https://auth.example.com/token",
        revoke_url="https://auth.example.com/revoke",
        token_endpoint_auth_method="none",
    )
    credentials = OAuth2Credentials(
        provider="mcp", access_token=SecretStr("<test-token>"), scopes=[]
    )
    with patch("backend.blocks.mcp.oauth.Requests") as requests:
        requests.return_value.post = AsyncMock(
            side_effect=HTTPClientError("Revocation rejected", 401)
        )
        assert await handler.revoke_tokens(credentials) is False
    requests.assert_called_once_with(raise_for_status=True)
