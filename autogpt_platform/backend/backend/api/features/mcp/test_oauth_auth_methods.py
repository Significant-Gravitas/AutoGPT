import base64
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id

from backend.api.features.mcp.routes import router
from backend.integrations.creds_manager import create_mcp_oauth_handler
from backend.util.request import HTTPClientError

app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: "test-user-id"


@pytest_asyncio.fixture(scope="module")
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def oauth_mocks():
    with (
        patch("backend.api.features.mcp.routes.MCPClient") as mcp_client,
        patch("backend.api.features.mcp.routes.creds_manager") as manager,
        patch("backend.api.features.mcp.routes.settings") as settings,
        patch("backend.api.features.mcp.routes.Requests") as requests,
        patch(
            "backend.api.features.mcp.routes.validate_url_host",
            new_callable=AsyncMock,
        ),
    ):
        metadata = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "revocation_endpoint": "https://auth.example.com/revoke",
            "registration_endpoint": "https://auth.example.com/register",
            "token_endpoint_auth_methods_supported": ["none"],
        }
        mcp_client.return_value.discover_auth = AsyncMock(return_value=None)
        mcp_client.return_value.discover_auth_server_metadata = AsyncMock(
            return_value=(metadata, "https://auth.example.com")
        )
        registration = {
            "client_id": "client:id",
            "client_secret": "<test secret>",
            "token_endpoint_auth_method": "none",
        }
        response = MagicMock()
        response.json.return_value = registration
        post = requests.return_value.post = AsyncMock(return_value=response)
        manager.store.store_state_token = AsyncMock(
            return_value=("state", "pkce-challenge")
        )
        manager.store.get_creds_by_provider = AsyncMock(return_value=[])
        manager.create = AsyncMock()
        settings.config.frontend_base_url = "https://app.example.com"
        yield metadata, registration, post, manager


@pytest.mark.parametrize(
    "advertised,requested,registered",
    [
        (["none"], "none", "none"),
        (["none"], "none", None),
        (["client_secret_basic"], "client_secret_basic", "client_secret_basic"),
        (["client_secret_post"], "client_secret_post", "client_secret_post"),
        (["client_secret_post", "none"], "client_secret_post", "none"),
        (None, "client_secret_basic", "client_secret_basic"),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_registered_method_survives_token_lifecycle(
    client, oauth_mocks, advertised, requested, registered
):
    metadata, registration, post, manager = oauth_mocks
    if advertised is None:
        metadata.pop("token_endpoint_auth_methods_supported")
    else:
        metadata["token_endpoint_auth_methods_supported"] = advertised
    if registered is None:
        registration.pop("token_endpoint_auth_method")
        registered = requested
    else:
        registration["token_endpoint_auth_method"] = registered

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.example.com/mcp"}
    )

    assert response.status_code == 200
    assert post.call_args.kwargs["json"]["token_endpoint_auth_method"] == requested
    state_metadata = manager.store.store_state_token.call_args.kwargs["state_metadata"]
    assert state_metadata["token_endpoint_auth_method"] == registered
    assert state_metadata["client_secret"] == (
        "" if registered == "none" else "<test secret>"
    )
    assert "client_secret" not in response.text
    state = MagicMock()
    state.state_metadata = state_metadata
    state.scopes = ["read"]
    state.code_verifier = "pkce-verifier"
    manager.store.verify_state_token = AsyncMock(return_value=state)

    with patch("backend.blocks.mcp.oauth.Requests") as token_requests:
        token_response = MagicMock()
        token_response.json.return_value = {
            "access_token": "<test-access-token>",
            "refresh_token": "<test-refresh-token>",
        }
        token_post = token_requests.return_value.post = AsyncMock(
            return_value=token_response
        )
        callback = await client.post(
            "/oauth/callback", json={"code": "code", "state_token": "state"}
        )
        assert callback.status_code == 200
        manager.store.verify_state_token.assert_awaited_once_with(
            "test-user-id", "state", "mcp"
        )
        exchange_args = token_post.call_args.kwargs
        assert exchange_args["data"]["code_verifier"] == "pkce-verifier"
        credentials = manager.create.call_args.args[1]
        assert credentials.metadata["mcp_token_endpoint_auth_method"] == registered
        assert credentials.metadata["mcp_client_secret"] == (
            "" if registered == "none" else "<test secret>"
        )
        assert "mcp_client_secret" not in callback.text
        await create_mcp_oauth_handler(credentials)._refresh_tokens(credentials)
        refresh_args = token_post.call_args.kwargs
        assert refresh_args["data"]["grant_type"] == "refresh_token"
        handler = create_mcp_oauth_handler(credentials)
        assert await handler.revoke_tokens(credentials)
        assert token_post.call_args.args[0] == "https://auth.example.com/revoke"
        revoke_args = token_post.call_args.kwargs
        assert revoke_args["data"]["token"] == "<test-access-token>"

    for args in (exchange_args, refresh_args, revoke_args):
        if registered == "client_secret_basic":
            assert (
                base64.b64decode(
                    args["headers"]["Authorization"].removeprefix("Basic ")
                ).decode()
                == "client%3Aid:%3Ctest+secret%3E"
            )
            assert "client_id" not in args["data"]
            assert "client_secret" not in args["data"]
        else:
            assert "Authorization" not in args["headers"]
            assert args["data"]["client_id"] == "client:id"
            if registered == "none":
                assert "client_secret" not in args["data"]
            else:
                assert args["data"]["client_secret"] == "<test secret>"


@pytest.mark.parametrize("unsupported_at", ["metadata", "registration"])
@pytest.mark.asyncio(loop_scope="session")
async def test_unsupported_method_stops_login(client, oauth_mocks, unsupported_at):
    metadata, registration, post, manager = oauth_mocks
    if unsupported_at == "metadata":
        metadata["token_endpoint_auth_methods_supported"] = ["private_key_jwt"]
    else:
        registration["token_endpoint_auth_method"] = "private_key_jwt"

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.example.com/mcp"}
    )

    assert response.status_code == 400
    assert "authentication method" in response.json()["detail"]
    manager.store.store_state_token.assert_not_awaited()
    if unsupported_at == "metadata":
        post.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_registered_secret_stops_login(client, oauth_mocks):
    metadata, registration, _, manager = oauth_mocks
    metadata["token_endpoint_auth_methods_supported"] = ["client_secret_post"]
    registration["token_endpoint_auth_method"] = "client_secret_post"
    registration.pop("client_secret")

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.example.com/mcp"}
    )

    assert response.status_code == 400
    assert "client secret" in response.json()["detail"]
    manager.store.store_state_token.assert_not_awaited()


@pytest.mark.parametrize(
    "failure", ["http_error", "missing_id", "empty_id", "invalid_url"]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_failed_registration_does_not_open_an_unregistered_login(
    client, oauth_mocks, failure
):
    _, registration, post, manager = oauth_mocks
    if failure == "http_error":
        post.side_effect = HTTPClientError("Registration denied", 403)
    elif failure == "missing_id":
        registration.pop("client_id")
    elif failure == "empty_id":
        registration["client_id"] = ""

    with patch(
        "backend.api.features.mcp.routes.validate_url_host",
        new_callable=AsyncMock,
    ) as validate:
        if failure == "invalid_url":
            validate.side_effect = [None, ValueError("Private host")]
        response = await client.post(
            "/oauth/login", json={"server_url": "https://mcp.example.com/mcp"}
        )

    assert response.status_code == 400
    assert "register" in response.json()["detail"].lower()
    manager.store.store_state_token.assert_not_awaited()
    if failure == "invalid_url":
        post.assert_not_awaited()


@pytest.mark.parametrize("scope_source", ["resource", "authorization_server"])
@pytest.mark.asyncio(loop_scope="session")
async def test_discovered_required_scopes_are_included(
    client, oauth_mocks, scope_source
):
    metadata, _, _, manager = oauth_mocks
    required_scopes = ["openid", "offline_access"]
    metadata["scopes_supported"] = (
        ["other_scope"] if scope_source == "resource" else required_scopes
    )
    resource = {
        "resource": "https://mcp.example.com/mcp",
        "authorization_servers": ["https://auth.example.com"],
    }
    if scope_source == "resource":
        resource["scopes_supported"] = required_scopes
    with patch("backend.api.features.mcp.routes.MCPClient") as mcp_client:
        mcp_client.return_value.discover_auth = AsyncMock(return_value=resource)
        mcp_client.return_value.discover_auth_server_metadata = AsyncMock(
            return_value=(metadata, "https://auth.example.com")
        )
        response = await client.post(
            "/oauth/login", json={"server_url": "https://mcp.example.com/mcp"}
        )

    assert response.status_code == 200
    assert parse_qs(urlparse(response.json()["login_url"]).query)["scope"] == [
        "openid offline_access"
    ]
    assert manager.store.store_state_token.call_args.args[2] == required_scopes
