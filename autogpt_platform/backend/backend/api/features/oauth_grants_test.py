"""The grant routes through the real app and database, signed in as two users."""

from typing import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id
from prisma.enums import APIKeyPermission
from prisma.models import OAuthApplication, User
from pydantic import BaseModel
from pytest_mock import MockerFixture

from backend.api.rest_api import app
from backend.data.auth.oauth import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    validate_access_token,
)
from backend.util.test import SpinTestServer

SCOPES = [APIKeyPermission.IDENTITY, APIKeyPermission.READ_GRAPH]


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
                "redirectUris": ["https://third-party.example.com/callback"],
                "scopes": SCOPES,
                "ownerId": developer_id,
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
        "backend.api.features.oauth_grants._push_shim_revocation",
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
async def test_another_user_revoking_the_app_ends_only_their_own_access(
    world: World, client: httpx.AsyncClient, shim_push: AsyncMock
):
    await _revoke_as(world.other_user_id, world, client)

    await validate_access_token(world.user_access_token)
    _sign_in_as(world.user_id)
    listed = (await client.get("/api/oauth/grants")).json()
    assert [grant["application_id"] for grant in listed] == [world.app_id]
    shim_push.assert_awaited_once_with(world.other_user_id, world.client_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_revoking_ends_the_apps_access_and_tells_the_local_pc(
    world: World, client: httpx.AsyncClient, shim_push: AsyncMock
):
    await _revoke_as(world.user_id, world, client)

    with pytest.raises(InvalidTokenError):
        await validate_access_token(world.user_access_token)
    await validate_access_token(world.other_user_access_token)
    shim_push.assert_awaited_once_with(world.user_id, world.client_id)
    assert (await client.get("/api/oauth/grants")).json() == []
    again = await client.delete(f"/api/oauth/grants/{world.app_id}")
    assert again.status_code == 404


async def _revoke_as(user_id: str, world: World, client: httpx.AsyncClient) -> None:
    _sign_in_as(user_id)
    response = await client.delete(f"/api/oauth/grants/{world.app_id}")
    assert response.status_code == 204


def _sign_in_as(user_id: str) -> None:
    app.dependency_overrides[get_user_id] = lambda: user_id
