"""Public clients, which registered themselves and have no secret: what they
must do instead, and what their existence must not loosen for other clients."""

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
from prisma.models import OAuthRefreshToken as PrismaOAuthRefreshToken

from backend.api.rest_api import app
from backend.data.auth.oauth import (
    cleanup_expired_oauth_tokens,
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
async def test_cleanup_deletes_only_self_registered_clients_nobody_used(
    client: httpx.AsyncClient,
):
    a_day_ago = datetime.now(UTC) - timedelta(days=1, minutes=1)
    unused = await _client(owner=None, created_at=a_day_ago)
    signed_in = await _client(owner=None, created_at=a_day_ago)
    await create_refresh_token(signed_in.id, USER_ID, [APIKeyPermission.READ_LIBRARY])
    fresh = await _client(owner=None, created_at=datetime.now(UTC))
    owned = await _client(owner=USER_ID, created_at=a_day_ago)
    kept = [signed_in, fresh, owned]
    try:
        await cleanup_expired_oauth_tokens()
        remaining = {
            row.id
            for row in await PrismaOAuthApplication.prisma().find_many(
                where={"id": {"in": [unused.id, *(c.id for c in kept)]}}
            )
        }
    finally:
        await PrismaOAuthRefreshToken.prisma().delete_many(
            where={"applicationId": signed_in.id}
        )
        await PrismaOAuthApplication.prisma().delete_many(
            where={"id": {"in": [unused.id, *(c.id for c in kept)]}}
        )

    assert remaining == {c.id for c in kept}


async def _authorize(
    client: httpx.AsyncClient, client_id: str, method: str, challenge: str = "x" * 43
) -> httpx.Response:
    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "scopes": SCOPES,
            "state": "state",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": method,
        },
    )
    assert response.status_code == 200, response.text
    return response


async def _client(owner: str | None, created_at: datetime) -> PrismaOAuthApplication:
    return await PrismaOAuthApplication.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": "Cleanup test",
            "clientId": f"test_client_{secrets.token_urlsafe(8)}",
            "redirectUris": [REDIRECT_URI],
            "scopes": [APIKeyPermission.READ_LIBRARY],
            "ownerId": owner,
            "createdAt": created_at,
        }
    )
