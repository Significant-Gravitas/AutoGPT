"""Sign-in to MCP servers that offer no Dynamic Client Registration.

The HubSpot metadata below is what mcp.hubspot.com served on 2026-10-10: no
``registration_endpoint`` and confidential clients only.
"""

from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id

from backend.api.features.mcp.routes import NO_OAUTH_CODE, router

app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: "test-user-id"

HUBSPOT_PROTECTED_RESOURCE = {
    "resource": "https://mcp.hubspot.com",
    "authorization_servers": ["https://mcp.hubspot.com"],
    "scopes_supported": [],
}
HUBSPOT_AUTH_SERVER = {
    "issuer": "https://mcp.hubspot.com",
    "authorization_endpoint": "https://mcp.hubspot.com/oauth/authorize/user",
    "token_endpoint": "https://mcp.hubspot.com/oauth/v3/token",
    "scopes_supported": [],
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "token_endpoint_auth_methods_supported": ["client_secret_post"],
    "code_challenge_methods_supported": ["S256"],
}


@pytest_asyncio.fixture(scope="module")
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def login():
    with (
        patch("backend.api.features.mcp.routes.MCPClient") as mcp_client,
        patch("backend.api.features.mcp.routes.creds_manager") as manager,
        patch("backend.api.features.mcp.routes.settings") as settings,
        patch(
            "backend.api.features.mcp.routes.validate_url_host",
            new_callable=AsyncMock,
        ),
    ):
        auth_server = dict(HUBSPOT_AUTH_SERVER)
        mcp_client.return_value.discover_auth = AsyncMock(
            return_value=HUBSPOT_PROTECTED_RESOURCE
        )
        mcp_client.return_value.discover_auth_server_metadata = AsyncMock(
            return_value=(auth_server, "https://mcp.hubspot.com")
        )
        manager.store.store_state_token = AsyncMock(
            return_value=("state-abc", "challenge-xyz")
        )
        settings.config.frontend_base_url = "https://platform.example.com"
        settings.secrets.hubspot_mcp_client_id = ""
        settings.secrets.hubspot_mcp_client_secret = ""
        yield auth_server, settings, manager


@pytest.mark.asyncio(loop_scope="session")
async def test_hubspot_signs_in_with_the_configured_app(client, login):
    _, settings, manager = login
    settings.secrets.hubspot_mcp_client_id = "hubspot-client-id"
    settings.secrets.hubspot_mcp_client_secret = "hubspot-secret"

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.hubspot.com"}
    )

    assert response.status_code == 200
    login_url = urlsplit(response.json()["login_url"])
    assert (login_url.scheme, login_url.hostname, login_url.path) == (
        "https",
        "mcp.hubspot.com",
        "/oauth/authorize/user",
    )
    query = parse_qs(login_url.query)
    assert query["client_id"] == ["hubspot-client-id"]
    assert query["code_challenge"] == ["challenge-xyz"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["resource"] == ["https://mcp.hubspot.com"]
    assert query["redirect_uri"] == [
        "https://platform.example.com/auth/integrations/mcp_callback"
    ]
    assert "hubspot-secret" not in login_url.query
    state = manager.store.store_state_token.call_args.kwargs["state_metadata"]
    assert state["client_id"] == "hubspot-client-id"
    assert state["client_secret"] == "hubspot-secret"
    assert state["token_endpoint_auth_method"] == "client_secret_post"
    assert state["token_url"] == "https://mcp.hubspot.com/oauth/v3/token"


@pytest.mark.asyncio(loop_scope="session")
async def test_hubspot_without_an_app_says_sign_in_is_not_set_up(client, login):
    _, _, manager = login

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.hubspot.com/"}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == NO_OAUTH_CODE
    assert detail["message"].startswith(
        "Sign-in to mcp.hubspot.com is not set up on this platform yet"
    )
    manager.store.store_state_token.assert_not_called()


@pytest.mark.parametrize(
    "methods", [["client_secret_post"], ["client_secret_basic", "private_key_jwt"]]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_unlisted_secret_only_server_is_refused_not_sent_a_placeholder(
    client, login, methods
):
    auth_server, _, manager = login
    auth_server["token_endpoint_auth_methods_supported"] = methods

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.vendor.example/mcp"}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == NO_OAUTH_CODE
    assert detail["message"] == (
        "Sign-in to mcp.vendor.example is not available on this platform: the "
        "server only accepts an OAuth app registered with it in advance, and "
        "none is registered for it here. "
        "You may need to provide an auth credential manually."
    )
    manager.store.store_state_token.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_cleartext_hubspot_url_is_refused_without_the_secret(client, login):
    _, settings, manager = login
    settings.secrets.hubspot_mcp_client_id = "hubspot-client-id"
    settings.secrets.hubspot_mcp_client_secret = "hubspot-secret"

    response = await client.post(
        "/oauth/login", json={"server_url": "http://mcp.hubspot.com"}
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == NO_OAUTH_CODE
    manager.store.store_state_token.assert_not_called()


@pytest.mark.parametrize("methods", [["none"], ["client_secret_post", "none"], None])
@pytest.mark.asyncio(loop_scope="session")
async def test_server_accepting_public_clients_keeps_the_placeholder(
    client, login, methods
):
    auth_server, _, manager = login
    if methods is None:
        auth_server.pop("token_endpoint_auth_methods_supported")
    else:
        auth_server["token_endpoint_auth_methods_supported"] = methods

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.vendor.example/mcp"}
    )

    assert response.status_code == 200
    query = parse_qs(urlsplit(response.json()["login_url"]).query)
    assert query["client_id"] == ["autogpt-platform"]
    state = manager.store.store_state_token.call_args.kwargs["state_metadata"]
    assert state["client_secret"] == ""
    assert state["token_endpoint_auth_method"] == "none"


# What mcp.withallo.com and api.withallo.com served on 2026-10-30: no
# ``registration_endpoint``, public clients allowed, and no scopes listed by
# the resource, so sign-in asks for every scope the server advertises.
ALLO_PROTECTED_RESOURCE = {
    "resource": "https://mcp.withallo.com/mcp",
    "authorization_servers": ["https://api.withallo.com"],
    "bearer_methods_supported": ["header"],
}
ALLO_SCOPES = ["CONVERSATIONS_READ", "SMS_SEND", "AGENTS_WRITE"]
ALLO_AUTH_SERVER = {
    "issuer": "https://api.withallo.com",
    "authorization_endpoint": "https://api.withallo.com/v1/oauth/authorize",
    "token_endpoint": "https://api.withallo.com/v1/oauth/token",
    "revocation_endpoint": "https://api.withallo.com/v1/oauth/revoke",
    "response_types_supported": ["code"],
    "grant_types_supported": [
        "authorization_code",
        "refresh_token",
        "client_credentials",
    ],
    "code_challenge_methods_supported": ["S256"],
    "client_id_metadata_document_supported": True,
    "authorization_response_iss_parameter_supported": True,
    "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    "scopes_supported": ALLO_SCOPES,
}


@pytest.fixture
def allo_login(login):
    _, settings, manager = login
    with patch("backend.api.features.mcp.routes.MCPClient") as mcp_client:
        mcp_client.return_value.discover_auth = AsyncMock(
            return_value=ALLO_PROTECTED_RESOURCE
        )
        mcp_client.return_value.discover_auth_server_metadata = AsyncMock(
            return_value=(dict(ALLO_AUTH_SERVER), "https://api.withallo.com")
        )
        settings.secrets.allo_mcp_client_id = ""
        yield settings, manager


@pytest.mark.asyncio(loop_scope="session")
async def test_allo_signs_in_with_its_public_app(client, allo_login):
    settings, manager = allo_login
    settings.secrets.allo_mcp_client_id = "allo-client-id"

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.withallo.com/mcp"}
    )

    assert response.status_code == 200
    login_url = urlsplit(response.json()["login_url"])
    assert (login_url.scheme, login_url.hostname, login_url.path) == (
        "https",
        "api.withallo.com",
        "/v1/oauth/authorize",
    )
    query = parse_qs(login_url.query)
    assert query["client_id"] == ["allo-client-id"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["resource"] == ["https://mcp.withallo.com/mcp"]
    assert query["scope"] == [" ".join(ALLO_SCOPES)]
    state = manager.store.store_state_token.call_args.kwargs["state_metadata"]
    assert state["client_id"] == "allo-client-id"
    assert state["client_secret"] == ""
    assert state["token_endpoint_auth_method"] == "none"
    assert state["token_url"] == "https://api.withallo.com/v1/oauth/token"
    assert state["revoke_url"] == "https://api.withallo.com/v1/oauth/revoke"
    assert state["iss_required"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_allo_without_an_app_says_sign_in_is_not_set_up(client, allo_login):
    _, manager = allo_login

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.withallo.com/mcp"}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == NO_OAUTH_CODE
    assert detail["message"].startswith(
        "Sign-in to mcp.withallo.com is not set up on this platform yet"
    )
    manager.store.store_state_token.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_hubspot_app_missing_its_secret_is_not_set_up(client, login):
    _, settings, manager = login
    settings.secrets.hubspot_mcp_client_id = "hubspot-client-id"

    response = await client.post(
        "/oauth/login", json={"server_url": "https://mcp.hubspot.com"}
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == NO_OAUTH_CODE
    manager.store.store_state_token.assert_not_called()
