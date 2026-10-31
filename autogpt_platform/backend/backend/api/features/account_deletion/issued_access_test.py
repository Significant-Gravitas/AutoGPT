"""revoke_issued_access: what AutoGPT issued on the user's behalf stops working."""

import secrets
from typing import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.api_key.keysmith import APIKeySmith
from prisma.enums import APIKeyPermission
from prisma.models import OAuthApplication, User
from pydantic import BaseModel
from pytest_mock import MockerFixture

from backend.api.external.fastapi_app import external_api
from backend.api.features.account_deletion.issued_access import revoke_issued_access
from backend.data.auth.api_key import create_api_key
from backend.data.auth.oauth import create_access_token, get_oauth_application_by_id
from backend.util.test import SpinTestServer

MODULE = "backend.api.features.account_deletion.issued_access"
SCOPES = [APIKeyPermission.IDENTITY]


# ---------------------------------------------------------------------------
# Unit: the two data helpers run for this user, and errors are not swallowed
# ---------------------------------------------------------------------------


@pytest.fixture
def revoke_keys(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{MODULE}.revoke_user_api_keys", return_value=2)


@pytest.fixture
def revoke_tokens(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.revoke_user_oauth_tokens",
        return_value={
            "authorization_codes": 1,
            "access_tokens": 3,
            "refresh_tokens": 3,
        },
    )


@pytest.mark.asyncio
async def test_revokes_the_users_api_keys_and_oauth_tokens(
    revoke_keys: AsyncMock, revoke_tokens: AsyncMock
):
    await revoke_issued_access("user-1")

    revoke_keys.assert_awaited_once_with("user-1")
    revoke_tokens.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["revoke_keys", "revoke_tokens"])
async def test_an_error_propagates_so_the_account_does_not_close(
    failing: str,
    request: pytest.FixtureRequest,
    revoke_keys: AsyncMock,
    revoke_tokens: AsyncMock,
):
    request.getfixturevalue(failing).side_effect = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        await revoke_issued_access("user-1")


# ---------------------------------------------------------------------------
# Integration: the external API rejects them on the very next request
# ---------------------------------------------------------------------------


class Holder(BaseModel):
    """One person's AutoGPT API key and OAuth access token (plaintext)."""

    user_id: str
    api_key: str
    access_token: str


class World(BaseModel):
    user: Holder
    bystander: Holder
    app_id: str


@pytest_asyncio.fixture(loop_scope="session")
async def world(server: SpinTestServer) -> AsyncIterator[World]:
    """The user also registered a third-party app as a developer, and the
    bystander authorized that app."""
    user_id, bystander_id = str(uuid4()), str(uuid4())
    await _settle_the_pool()
    try:
        for uid in (user_id, bystander_id):
            await User.prisma().create(
                data={"id": uid, "email": f"issued-access-{uid}@example.com"}
            )
        app_id = await _register_app(owner_id=user_id)
        yield World(
            user=await _issue(user_id, app_id),
            bystander=await _issue(bystander_id, app_id),
            app_id=app_id,
        )
    finally:
        await User.prisma().delete_many(where={"id": {"in": [user_id, bystander_id]}})


@pytest_asyncio.fixture(loop_scope="session")
async def external_client(world: World) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=external_api), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio(loop_scope="session")
async def test_the_next_external_api_request_is_rejected(
    world: World, external_client: httpx.AsyncClient
):
    """Keys and tokens are checked against the database on every request, so
    nothing cached keeps a revoked one working."""
    assert await _status(external_client, world.user) == (200, 200)

    await revoke_issued_access(world.user.user_id)

    assert await _status(external_client, world.user) == (401, 401)
    assert await _status(external_client, world.bystander) == (200, 200)
    app = await get_oauth_application_by_id(world.app_id)
    assert app is not None and app.is_active


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_run_is_harmless(
    world: World, external_client: httpx.AsyncClient
):
    await revoke_issued_access(world.user.user_id)
    await revoke_issued_access(world.user.user_id)

    assert await _status(external_client, world.user) == (401, 401)
    assert await _status(external_client, world.bystander) == (200, 200)


async def _status(client: httpx.AsyncClient, holder: Holder) -> tuple[int, int]:
    by_key = await client.get("/v1/me", headers={"X-API-Key": holder.api_key})
    by_token = await client.get(
        "/v1/me", headers={"Authorization": f"Bearer {holder.access_token}"}
    )
    return by_key.status_code, by_token.status_code


async def _issue(user_id: str, app_id: str) -> Holder:
    _, api_key = await create_api_key(
        name="integration", user_id=user_id, permissions=SCOPES
    )
    token = await create_access_token(app_id, user_id, SCOPES)
    return Holder(
        user_id=user_id,
        api_key=api_key,
        access_token=token.token.get_secret_value(),
    )


async def _register_app(owner_id: str) -> str:
    secret_hash, secret_salt = APIKeySmith().hash_key(
        f"agpt_secret_{secrets.token_urlsafe(16)}"
    )
    app = await OAuthApplication.prisma().create(
        data={
            "name": "The user's own app",
            "clientId": f"client-{uuid4()}",
            "clientSecret": secret_hash,
            "clientSecretSalt": secret_salt,
            "redirectUris": ["https://app.example.com/callback"],
            "scopes": SCOPES,
            "ownerId": owner_id,
        }
    )
    return app.id


async def _settle_the_pool() -> None:
    """An earlier test can leave the Prisma pool bound to a closed event loop;
    only the first query on the session loop after that fails."""
    try:
        await User.prisma().count(where={"id": "pool-check"})
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
