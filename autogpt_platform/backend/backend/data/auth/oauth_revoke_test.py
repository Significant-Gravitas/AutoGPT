"""Revoking every OAuth grant AutoGPT issued for a user, against the real database."""

from datetime import datetime
from typing import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import APIKeyPermission
from prisma.models import OAuthAccessToken, OAuthApplication
from prisma.models import OAuthAuthorizationCode as PrismaAuthorizationCode
from prisma.models import OAuthRefreshToken, User
from pydantic import BaseModel

from backend.data.auth.oauth import (
    InvalidGrantError,
    InvalidTokenError,
    consume_authorization_code,
    create_access_token,
    create_authorization_code,
    create_refresh_token,
    get_oauth_application_by_id,
    refresh_tokens,
    revoke_user_oauth_tokens,
    validate_access_token,
)
from backend.util.test import SpinTestServer

REDIRECT_URI = "https://third-party.example.com/callback"
SCOPES = [APIKeyPermission.IDENTITY]


class Grants(BaseModel):
    """What one user handed one third-party app: plaintext tokens and code."""

    access_token: str
    refresh_token: str
    unused_code: str
    used_code: str


class World(BaseModel):
    user_id: str
    bystander_id: str
    app_id: str


async def _settle_the_pool() -> None:
    """An earlier test can leave the Prisma pool bound to a closed event loop;
    only the first query on the session loop after that fails."""
    try:
        await User.prisma().count(where={"id": "pool-check"})
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise


@pytest_asyncio.fixture(loop_scope="session")
async def world(server: SpinTestServer) -> AsyncIterator[World]:
    """A user, a bystander, and a third-party app a developer registered."""
    user_id, bystander_id, developer_id = (str(uuid4()) for _ in range(3))
    await _settle_the_pool()
    try:
        for uid in (user_id, bystander_id, developer_id):
            await User.prisma().create(
                data={"id": uid, "email": f"oauth-revoke-{uid}@example.com"}
            )
        app = await OAuthApplication.prisma().create(
            data={
                "name": "Third-party app",
                "clientId": f"client-{uuid4()}",
                "clientSecret": "not-used-here",
                "clientSecretSalt": "not-used-here",
                "redirectUris": [REDIRECT_URI],
                "scopes": SCOPES,
                "ownerId": developer_id,
            }
        )
        yield World(user_id=user_id, bystander_id=bystander_id, app_id=app.id)
    finally:
        # Tokens, codes and the app cascade with their users.
        await User.prisma().delete_many(
            where={"id": {"in": [user_id, bystander_id, developer_id]}}
        )


async def _grant(app_id: str, user_id: str) -> Grants:
    access = await create_access_token(app_id, user_id, SCOPES)
    refresh = await create_refresh_token(app_id, user_id, SCOPES)
    unused = await create_authorization_code(app_id, user_id, SCOPES, REDIRECT_URI)
    used = await create_authorization_code(app_id, user_id, SCOPES, REDIRECT_URI)
    await consume_authorization_code(used.code, app_id, REDIRECT_URI)
    return Grants(
        access_token=access.token.get_secret_value(),
        refresh_token=refresh.token.get_secret_value(),
        unused_code=unused.code,
        used_code=used.code,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_every_token_and_unused_code_of_the_user_is_revoked(world: World):
    user = await _grant(world.app_id, world.user_id)
    bystander = await _grant(world.app_id, world.bystander_id)

    counts = await revoke_user_oauth_tokens(world.user_id)

    assert counts == {"authorization_codes": 1, "access_tokens": 1, "refresh_tokens": 1}
    with pytest.raises(InvalidTokenError, match="revoked"):
        await validate_access_token(user.access_token)
    with pytest.raises(InvalidGrantError, match="revoked"):
        await refresh_tokens(user.refresh_token, world.app_id)
    with pytest.raises(InvalidGrantError):
        await consume_authorization_code(user.unused_code, world.app_id, REDIRECT_URI)
    codes = await PrismaAuthorizationCode.prisma().find_many(
        where={"userId": world.user_id}
    )
    assert [code.code for code in codes] == [user.used_code]
    await _assert_untouched(bystander, world)
    app = await get_oauth_application_by_id(world.app_id)
    assert app is not None and app.is_active


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_run_revokes_nothing_and_keeps_the_revocation_time(
    world: World,
):
    await _grant(world.app_id, world.user_id)
    await revoke_user_oauth_tokens(world.user_id)
    first = await _revoked_at(world.user_id)
    assert first and None not in first

    counts = await revoke_user_oauth_tokens(world.user_id)

    assert counts == {"authorization_codes": 0, "access_tokens": 0, "refresh_tokens": 0}
    assert await _revoked_at(world.user_id) == first


@pytest.mark.asyncio(loop_scope="session")
async def test_a_user_without_grants_revokes_nothing(world: World):
    counts = await revoke_user_oauth_tokens(world.user_id)

    assert counts == {"authorization_codes": 0, "access_tokens": 0, "refresh_tokens": 0}


async def _assert_untouched(grants: Grants, world: World) -> None:
    token, _ = await validate_access_token(grants.access_token)
    assert token.user_id == world.bystander_id
    rows = await OAuthRefreshToken.prisma().find_many(
        where={"userId": world.bystander_id}
    )
    assert [row.revokedAt for row in rows] == [None]
    codes = await PrismaAuthorizationCode.prisma().find_many(
        where={"userId": world.bystander_id}
    )
    assert {code.code for code in codes} == {grants.unused_code, grants.used_code}


async def _revoked_at(user_id: str) -> list[datetime | None]:
    access = await OAuthAccessToken.prisma().find_many(
        where={"userId": user_id}, order={"id": "asc"}
    )
    refresh = await OAuthRefreshToken.prisma().find_many(
        where={"userId": user_id}, order={"id": "asc"}
    )
    return [row.revokedAt for row in [*access, *refresh]]
