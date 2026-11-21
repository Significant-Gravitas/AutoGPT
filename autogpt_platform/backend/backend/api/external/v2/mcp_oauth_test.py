"""An MCP client signs in to the MCP server with OAuth, from the 401 alone.

Driven by the MCP SDK's own OAuth client against the real app and database:
it discovers the provider from the 401, registers itself if it has no client
ID, sends the user to consent, redeems the code at the token endpoint the way
RFC 6749 clients do, and then calls the MCP server with the token it got.
"""

import secrets
import uuid
from collections.abc import AsyncGenerator
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.api_key.keysmith import APIKeySmith
from autogpt_libs.auth import get_user_id
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from prisma.enums import APIKeyPermission
from prisma.models import OAuthApplication as PrismaOAuthApplication

from backend.api.external.v2.mcp_server import (
    mcp_mount,
    mcp_scopes,
    protected_resource_metadata,
)
from backend.api.features import oauth_registration
from backend.api.features.oauth_metadata import AUTHORIZE_PAGE_PATH
from backend.api.rest_api import app
from backend.data.user import get_or_create_user_with_status

# Fixed, so reruns reuse one user instead of adding one per run.
USER_ID = "3c1a6c0e-6b51-4f0e-9a51-7d1c2f9e4a10"
REDIRECT_URI = "http://localhost:8765/callback"
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


@pytest_asyncio.fixture(loop_scope="session")
async def mcp_user(server) -> str:
    await get_or_create_user_with_status(
        {"sub": USER_ID, "email": "mcp-oauth-test@example.com"}
    )
    return USER_ID


@pytest_asyncio.fixture(loop_scope="session")
async def oauth_app(mcp_user: str) -> AsyncGenerator[OAuthClientInformationFull, None]:
    """A pre-registered app allowed every scope the MCP server asks for."""
    client_id = f"test_client_{secrets.token_urlsafe(8)}"
    client_secret = f"agpt_secret_{secrets.token_urlsafe(16)}"
    secret_hash, secret_salt = APIKeySmith().hash_key(client_secret)
    app_row = await PrismaOAuthApplication.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": "MCP OAuth test client",
            "clientId": client_id,
            "clientSecret": secret_hash,
            "clientSecretSalt": secret_salt,
            "redirectUris": [REDIRECT_URI],
            "scopes": [APIKeyPermission(s) for s in mcp_scopes()],
            "ownerId": USER_ID,
        }
    )
    yield OAuthClientInformationFull(
        client_id=client_id, client_secret=client_secret, redirect_uris=[REDIRECT_URI]
    )
    await PrismaOAuthApplication.prisma().delete(where={"id": app_row.id})


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("auth_method", ["client_secret_basic", "client_secret_post"])
async def test_an_mcp_client_signs_in_with_a_pre_registered_app(
    oauth_app: OAuthClientInformationFull, auth_method: str
):
    storage = _Storage(
        oauth_app.model_copy(update={"token_endpoint_auth_method": auth_method})
    )
    consent = await _sign_in(storage, OAuthClientMetadata(redirect_uris=[REDIRECT_URI]))

    assert urlparse(consent.authorization_url).path == AUTHORIZE_PAGE_PATH
    assert storage.tokens is not None
    assert set((storage.tokens.scope or "").split()) == set(mcp_scopes())


@pytest.mark.asyncio(loop_scope="session")
async def test_an_mcp_client_registers_itself_and_signs_in(
    mcp_user: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        oauth_registration.settings.config, "oauth_dynamic_client_registration", True
    )
    storage = _Storage(None)
    try:
        await _sign_in(
            storage,
            OAuthClientMetadata(
                redirect_uris=[REDIRECT_URI],
                token_endpoint_auth_method="none",
                client_name="MCP test client",
            ),
        )
    finally:
        if storage.client_info:
            await PrismaOAuthApplication.prisma().delete_many(
                where={"clientId": storage.client_info.client_id}
            )

    assert storage.client_info is not None
    assert storage.client_info.client_secret is None
    assert storage.tokens is not None
    assert set((storage.tokens.scope or "").split()) == set(mcp_scopes())


async def _sign_in(
    storage: "_Storage", client_metadata: OAuthClientMetadata
) -> "_ConsentPage":
    """Call the MCP server as a signed-out client; it must sign in to get a 200."""
    server_url = str(protected_resource_metadata().resource)
    consent = _ConsentPage()
    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=consent.open,
        callback_handler=consent.callback,
    )
    app.dependency_overrides[get_user_id] = lambda: USER_ID
    try:
        async with mcp_mount.running(), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), auth=provider
        ) as client:
            response = await client.post(
                f"{server_url}/",
                json=INITIALIZE,
                headers={"Accept": "application/json, text/event-stream"},
            )
    finally:
        app.dependency_overrides.pop(get_user_id, None)

    assert response.status_code == 200, response.text
    assert "autogpt-platform" in response.text
    return consent


class _ConsentPage:
    """What the frontend's consent page does when the user approves."""

    def __init__(self) -> None:
        self.authorization_url = ""
        self._redirect_url = ""

    async def open(self, authorization_url: str) -> None:
        self.authorization_url = authorization_url
        query = {
            k: v[0] for k, v in parse_qs(urlparse(authorization_url).query).items()
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/oauth/authorize",
                json={
                    "client_id": query["client_id"],
                    "redirect_uri": query["redirect_uri"],
                    "scopes": query["scope"].split(),
                    "state": query["state"],
                    "response_type": query["response_type"],
                    "code_challenge": query["code_challenge"],
                    "code_challenge_method": query["code_challenge_method"],
                },
            )
        assert response.status_code == 200, response.text
        self._redirect_url = response.json()["redirect_url"]

    async def callback(self) -> tuple[str, str | None]:
        query = parse_qs(urlparse(self._redirect_url).query)
        return query["code"][0], query["state"][0]


class _Storage:
    def __init__(self, client_info: OAuthClientInformationFull | None) -> None:
        self.client_info = client_info
        self.tokens: OAuthToken | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info
