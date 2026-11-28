"""The grant routes through the real app and database, signed in as two users."""

import base64
import hashlib
import secrets
from typing import AsyncIterator
from unittest.mock import AsyncMock, call
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id
from prisma.enums import APIKeyPermission
from prisma.models import OAuthAccessToken, OAuthApplication, OAuthRefreshToken, User
from pydantic import BaseModel
from pytest_mock import MockerFixture

import backend.api.features.oauth as oauth_routes
from backend.api.rest_api import app
from backend.data.auth.oauth import (
    InvalidTokenError,
    create_access_token,
    create_authorization_code,
    create_refresh_token,
    validate_access_token,
)
from backend.data.auth.oauth_grants import revoke_user_oauth_grant
from backend.util.test import SpinTestServer

SCOPES = [APIKeyPermission.IDENTITY, APIKeyPermission.READ_GRAPH]
REDIRECT_URI = "https://third-party.example.com/callback"


class World(BaseModel):
    user_id: str
    other_user_id: str
    app_id: str
    client_id: str
    user_access_token: str
    other_user_access_token: str


@pytest_asyncio.fixture(loop_scope="session")
async def world(server: SpinTestServer) -> AsyncIterator[World]:
    """Two users who each signed in to the same app."""
    user_id, other_user_id, developer_id = (str(uuid4()) for _ in range(3))
    try:
        for uid in (user_id, other_user_id, developer_id):
            await User.prisma().create(
                data={"id": uid, "email": f"oauth-grant-routes-{uid}@example.com"}
            )
        oauth_app = await OAuthApplication.prisma().create(
            data={
                "name": "Notes app",
                "clientId": f"client-{uuid4()}",
                "clientSecret": "not-used-here",
                "clientSecretSalt": "not-used-here",
                "redirectUris": [REDIRECT_URI],
                "scopes": SCOPES,
                "ownerId": developer_id,
                "isPublic": True,
            }
        )
        tokens = []
        for uid in (user_id, other_user_id):
            await create_refresh_token(oauth_app.id, uid, SCOPES)
            access = await create_access_token(oauth_app.id, uid, SCOPES)
            tokens.append(access.token.get_secret_value())
        yield World(
            user_id=user_id,
            other_user_id=other_user_id,
            app_id=oauth_app.id,
            client_id=oauth_app.clientId,
            user_access_token=tokens[0],
            other_user_access_token=tokens[1],
        )
    finally:
        await User.prisma().delete_many(
            where={"id": {"in": [user_id, other_user_id, developer_id]}}
        )


@pytest_asyncio.fixture(loop_scope="session")
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client
    app.dependency_overrides.pop(get_user_id, None)


@pytest.fixture
def shim_push(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        "backend.api.features.oauth_grants.push_shim_revocation",
        new_callable=AsyncMock,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_the_user_sees_the_app_with_what_it_may_do(
    world: World, client: httpx.AsyncClient
):
    _sign_in_as(world.user_id)

    response = await client.get("/api/oauth/grants")

    assert response.status_code == 200
    [grant] = response.json()
    assert grant["application_id"] == world.app_id
    assert grant["name"] == "Notes app"
    assert grant["scopes"] == ["IDENTITY", "READ_GRAPH"]


@pytest.mark.asyncio(loop_scope="session")
async def test_each_user_revokes_only_their_own_access(
    world: World, client: httpx.AsyncClient, shim_push: AsyncMock
):
    await _revoke_as(world.other_user_id, world, client)
    await validate_access_token(world.user_access_token)

    await _revoke_as(world.user_id, world, client)

    with pytest.raises(InvalidTokenError):
        await validate_access_token(world.user_access_token)
    assert shim_push.await_args_list == [
        call(world.other_user_id, world.client_id),
        call(world.user_id, world.client_id),
    ]
    assert (await client.get("/api/oauth/grants")).json() == []
    again = await client.delete(f"/api/oauth/grants/{world.app_id}")
    assert again.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
async def test_a_code_exchange_the_revoke_overtakes_gets_no_tokens(
    world: World, client: httpx.AsyncClient, mocker: MockerFixture
):
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    code = await create_authorization_code(
        world.app_id,
        world.user_id,
        SCOPES,
        REDIRECT_URI,
        code_challenge=challenge.decode().rstrip("="),
        code_challenge_method="S256",
    )
    mint = oauth_routes.create_refresh_token

    async def revoke_after_the_claim(*args, **kwargs):
        await revoke_user_oauth_grant(world.user_id, world.app_id)
        return await mint(*args, **kwargs)

    mocker.patch.object(oauth_routes, "create_refresh_token", revoke_after_the_claim)

    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": code.code,
            "redirect_uri": REDIRECT_URI,
            "client_id": world.client_id,
            "code_verifier": verifier,
        },
    )

    assert response.status_code == 400
    live = {"userId": world.user_id, "revokedAt": None}
    assert await OAuthAccessToken.prisma().count(where=live) == 0
    assert await OAuthRefreshToken.prisma().count(where=live) == 0


async def _revoke_as(user_id: str, world: World, client: httpx.AsyncClient) -> None:
    _sign_in_as(user_id)
    response = await client.delete(f"/api/oauth/grants/{world.app_id}")
    assert response.status_code == 204


def _sign_in_as(user_id: str) -> None:
    app.dependency_overrides[get_user_id] = lambda: user_id
