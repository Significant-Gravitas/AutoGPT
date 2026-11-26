"""Public clients, which registered themselves and have no secret: what they
must do instead, what they may be granted, and what their existence must not
loosen for other clients."""

import secrets
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.api_key.keysmith import APIKeySmith
from autogpt_libs.auth import get_user_id
from prisma.enums import APIKeyPermission
from prisma.models import OAuthApplication as PrismaOAuthApplication

from backend.api.external.v2.mcp_scopes import mcp_scopes
from backend.api.features import oauth_registration
from backend.api.rest_api import app
from backend.data.auth.oauth import (
    cleanup_expired_oauth_tokens,
    create_access_token,
    create_authorization_code,
    create_refresh_token,
    register_public_client,
)
from backend.data.user import get_or_create_user_with_status

USER_ID = "8f2d0b6a-5c3e-4d71-b9a4-2e6f1c7d3b58"
REDIRECT_URI = "http://localhost:8765/callback"
SCOPES = ["READ_LIBRARY"]


@pytest_asyncio.fixture(loop_scope="session")
async def client(server) -> AsyncGenerator[httpx.AsyncClient, None]:
    await get_or_create_user_with_status(
        {"sub": USER_ID, "email": "oauth-public-client-test@example.com"}
    )
    app.dependency_overrides[get_user_id] = lambda: USER_ID
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            yield http_client
    finally:
        app.dependency_overrides.pop(get_user_id, None)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_public_client_must_use_s256(client: httpx.AsyncClient):
    public = await register_public_client(
        name="Public",
        redirect_uris=[REDIRECT_URI],
        grant_types=["authorization_code"],
        scopes=[APIKeyPermission(s) for s in SCOPES],
    )
    try:
        response = await _authorize(client, public.client_id, method="plain")
    finally:
        await PrismaOAuthApplication.prisma().delete(where={"id": public.id})

    query = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert query["error"] == ["invalid_request"]
    assert "code" not in query


@pytest.mark.asyncio(loop_scope="session")
async def test_a_client_that_names_no_scope_gets_the_mcp_server_permissions(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    fresh_registration_limits: None,
):
    monkeypatch.setattr(
        oauth_registration.settings.config, "oauth_dynamic_client_registration", True
    )
    response = await client.post(
        "/api/oauth/register", json={"redirect_uris": [REDIRECT_URI]}
    )
    assert response.status_code == 201, response.text
    client_id = response.json()["client_id"]
    try:
        stored = await PrismaOAuthApplication.prisma().find_unique(
            where={"clientId": client_id}
        )
    finally:
        await PrismaOAuthApplication.prisma().delete_many(where={"clientId": client_id})

    assert set(response.json()["scope"].split()) == set(mcp_scopes())
    assert stored is not None and set(stored.scopes) == set(mcp_scopes())


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("owner, granted", [(None, False), (USER_ID, True)])
async def test_only_a_self_registered_client_is_held_to_the_mcp_permissions(
    client: httpx.AsyncClient, owner: str | None, granted: bool
):
    """Its row can hold more: it registered before a tool stopped needing one."""
    assert "MANAGE_INTEGRATIONS" not in mcp_scopes()
    scopes = ["READ_LIBRARY", "MANAGE_INTEGRATIONS"]
    registered = await _client(owner=owner, created_at=datetime.now(UTC), scopes=scopes)
    try:
        response = await _authorize(
            client, registered.clientId, method="S256", scopes=scopes
        )
    finally:
        await PrismaOAuthApplication.prisma().delete(where={"id": registered.id})

    query = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert ("code" in query) == granted
    assert query.get("error") == (None if granted else ["invalid_scope"])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_confidential_client_still_needs_its_secret(
    client: httpx.AsyncClient,
):
    """The secret became optional for public clients, not for anyone else."""
    client_id = f"test_client_{secrets.token_urlsafe(8)}"
    secret_hash, salt = APIKeySmith().hash_key(
        f"agpt_secret_{secrets.token_urlsafe(16)}"
    )
    confidential = await PrismaOAuthApplication.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": "Confidential",
            "clientId": client_id,
            "clientSecret": secret_hash,
            "clientSecretSalt": salt,
            "redirectUris": [REDIRECT_URI],
            "scopes": [APIKeyPermission(s) for s in SCOPES],
            "ownerId": USER_ID,
        }
    )
    verifier = secrets.token_urlsafe(48)
    try:
        authorized = await _authorize(
            client, client_id, method="plain", challenge=verifier
        )
        code = parse_qs(urlparse(authorized.json()["redirect_url"]).query)["code"][0]
        response = await client.post(
            "/api/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
    finally:
        await PrismaOAuthApplication.prisma().delete(where={"id": confidential.id})

    assert response.status_code == 401, response.text
    assert "access_token" not in response.text


@pytest.mark.asyncio(loop_scope="session")
async def test_cleanup_deletes_a_self_registered_client_after_30_idle_days(
    client: httpx.AsyncClient,
):
    now = datetime.now(UTC)
    long_ago = now - timedelta(days=60)
    idle_since = now - timedelta(days=30, minutes=1)
    idle = await _client(owner=None, created_at=long_ago, last_used=idle_since)
    recently_used = await _client(
        owner=None, created_at=long_ago, last_used=now - timedelta(days=29, hours=23)
    )
    signed_in = await _client(owner=None, created_at=long_ago, last_used=idle_since)
    await create_refresh_token(signed_in.id, USER_ID, [APIKeyPermission.READ_LIBRARY])
    signing_in = await _client(owner=None, created_at=long_ago, last_used=idle_since)
    await create_authorization_code(
        signing_in.id, USER_ID, [APIKeyPermission.READ_LIBRARY], REDIRECT_URI
    )
    # Issuing the code counted as use; age it again so only the code keeps it.
    await PrismaOAuthApplication.prisma().update(
        where={"id": signing_in.id}, data={"lastUsedAt": idle_since}
    )
    owned = await _client(owner=USER_ID, created_at=long_ago, last_used=idle_since)
    kept = [recently_used, signed_in, signing_in, owned]
    try:
        await cleanup_expired_oauth_tokens()
        remaining = {
            row.id
            for row in await PrismaOAuthApplication.prisma().find_many(
                where={"id": {"in": [idle.id, *(c.id for c in kept)]}}
            )
        }
    finally:
        await PrismaOAuthApplication.prisma().delete_many(
            where={"id": {"in": [idle.id, *(c.id for c in kept)]}}
        )

    assert remaining == {c.id for c in kept}


ISSUES = {
    "authorization": lambda app_id: create_authorization_code(
        app_id, USER_ID, [APIKeyPermission.READ_LIBRARY], REDIRECT_URI
    ),
    "token": lambda app_id: create_access_token(
        app_id, USER_ID, [APIKeyPermission.READ_LIBRARY]
    ),
}


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("issue", ISSUES)
async def test_an_authorization_or_a_token_counts_as_use(
    client: httpx.AsyncClient, issue: str
):
    """A refresh issues a token too, so a client in use never goes idle."""
    long_ago = datetime.now(UTC) - timedelta(days=60)
    registered = await _client(owner=None, created_at=long_ago, last_used=long_ago)
    try:
        await ISSUES[issue](registered.id)
        stored = await PrismaOAuthApplication.prisma().find_unique(
            where={"id": registered.id}
        )
    finally:
        await PrismaOAuthApplication.prisma().delete(where={"id": registered.id})

    assert stored is not None
    assert stored.lastUsedAt > datetime.now(UTC) - timedelta(minutes=5)


async def _authorize(
    client: httpx.AsyncClient,
    client_id: str,
    method: str,
    challenge: str = "x" * 43,
    scopes: list[str] = SCOPES,
) -> httpx.Response:
    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "scopes": scopes,
            "state": "state",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": method,
        },
    )
    assert response.status_code == 200, response.text
    return response


async def _client(
    owner: str | None,
    created_at: datetime,
    scopes: list[str] = SCOPES,
    last_used: datetime | None = None,
) -> PrismaOAuthApplication:
    return await PrismaOAuthApplication.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": "Public client test",
            "clientId": f"test_client_{secrets.token_urlsafe(8)}",
            "redirectUris": [REDIRECT_URI],
            "scopes": [APIKeyPermission(s) for s in scopes],
            "ownerId": owner,
            "createdAt": created_at,
            "lastUsedAt": last_used or created_at,
        }
    )
