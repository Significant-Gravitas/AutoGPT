"""Listing and revoking one user's OAuth grants, against the real database."""

from datetime import timedelta
from typing import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import APIKeyPermission
from prisma.models import OAuthAccessToken, OAuthApplication
from prisma.models import OAuthAuthorizationCode as PrismaAuthorizationCode
from prisma.models import OAuthRefreshToken, User
from pydantic import BaseModel

from backend.data.auth import oauth
from backend.data.auth.oauth import (
    InvalidGrantError,
    InvalidTokenError,
    create_access_token,
    create_authorization_code,
    create_refresh_token,
    refresh_tokens,
    validate_access_token,
)
from backend.data.auth.oauth_grants import (
    list_user_oauth_grants,
    revoke_user_oauth_grant,
)
from backend.util.test import SpinTestServer

REDIRECT_URI = "https://third-party.example.com/callback"


class Grant(BaseModel):
    """One sign-in of an app: the token pair the token endpoint hands out."""

    access_token: str
    refresh_token: str


class World(BaseModel):
    user_id: str
    bystander_id: str
    app_id: str
    other_app_id: str


@pytest_asyncio.fixture(loop_scope="session")
async def world(server: SpinTestServer) -> AsyncIterator[World]:
    """A user, a bystander, and two apps a developer registered."""
    user_id, bystander_id, developer_id = (str(uuid4()) for _ in range(3))
    try:
        for uid in (user_id, bystander_id, developer_id):
            await User.prisma().create(
                data={"id": uid, "email": f"oauth-grants-{uid}@example.com"}
            )
        app_ids = []
        for name in ("Notes app", "Calendar app"):
            app = await OAuthApplication.prisma().create(
                data={
                    "name": name,
                    "clientId": f"client-{uuid4()}",
                    "clientSecret": "not-used-here",
                    "clientSecretSalt": "not-used-here",
                    "redirectUris": [REDIRECT_URI],
                    "scopes": list(APIKeyPermission),
                    "ownerId": developer_id,
                }
            )
            app_ids.append(app.id)
        yield World(
            user_id=user_id,
            bystander_id=bystander_id,
            app_id=app_ids[0],
            other_app_id=app_ids[1],
        )
    finally:
        # Tokens, codes and the apps cascade with their users.
        await User.prisma().delete_many(
            where={"id": {"in": [user_id, bystander_id, developer_id]}}
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_each_app_is_listed_once_with_every_permission_its_sign_ins_hold(
    world: World,
):
    await _sign_in(world.app_id, world.user_id, [APIKeyPermission.READ_GRAPH])
    newest = await _sign_in(world.app_id, world.user_id, [APIKeyPermission.IDENTITY])
    await _sign_in(world.other_app_id, world.user_id, [APIKeyPermission.READ_RUN])
    await _sign_in(world.other_app_id, world.bystander_id, [APIKeyPermission.USE_TOOLS])
    await _age_tokens(world.other_app_id, world.user_id, timedelta(minutes=5))

    grants = await list_user_oauth_grants(world.user_id)

    assert [(g.application_id, g.name, g.scopes) for g in grants] == [
        (
            world.app_id,
            "Notes app",
            [APIKeyPermission.IDENTITY, APIKeyPermission.READ_GRAPH],
        ),
        (world.other_app_id, "Calendar app", [APIKeyPermission.READ_RUN]),
    ]
    issued = await OAuthAccessToken.prisma().find_unique_or_raise(
        where={"token": oauth._hash_token(newest.access_token)}
    )
    assert grants[0].last_active_at == issued.createdAt


@pytest.mark.asyncio(loop_scope="session")
async def test_an_app_whose_tokens_are_all_revoked_or_expired_is_not_listed(
    world: World,
):
    await _sign_in(world.app_id, world.user_id, [APIKeyPermission.IDENTITY])
    await revoke_user_oauth_grant(world.user_id, world.app_id)
    await _sign_in(world.other_app_id, world.user_id, [APIKeyPermission.IDENTITY])
    await _age_tokens(world.other_app_id, world.user_id, timedelta(days=31))

    assert await list_user_oauth_grants(world.user_id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_revoking_ends_only_that_apps_access_for_that_user(world: World):
    scopes = [APIKeyPermission.IDENTITY]
    revoked = [await _sign_in(world.app_id, world.user_id, scopes) for _ in range(2)]
    unused_code = await create_authorization_code(
        world.app_id, world.user_id, scopes, REDIRECT_URI
    )
    other_app = await _sign_in(world.other_app_id, world.user_id, scopes)
    bystander = await _sign_in(world.app_id, world.bystander_id, scopes)

    assert await revoke_user_oauth_grant(world.user_id, world.app_id) == 4

    for grant in revoked:
        with pytest.raises(InvalidTokenError):
            await validate_access_token(grant.access_token)
        with pytest.raises(InvalidGrantError):
            await refresh_tokens(grant.refresh_token, world.app_id)
    assert not await PrismaAuthorizationCode.prisma().find_unique(
        where={"code": unused_code.code}
    )
    await validate_access_token(other_app.access_token)
    await validate_access_token(bystander.access_token)
    await refresh_tokens(bystander.refresh_token, world.app_id)
    assert [g.application_id for g in await list_user_oauth_grants(world.user_id)] == [
        world.other_app_id
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_app_with_no_access_for_the_user_revokes_nothing(world: World):
    bystander = await _sign_in(
        world.app_id, world.bystander_id, [APIKeyPermission.IDENTITY]
    )

    assert await revoke_user_oauth_grant(world.user_id, world.app_id) == 0
    assert await revoke_user_oauth_grant(world.user_id, str(uuid4())) == 0

    await validate_access_token(bystander.access_token)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_refresh_rotating_while_the_user_revokes_leaves_no_live_token(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    grant = await _sign_in(world.app_id, world.user_id, [APIKeyPermission.IDENTITY])
    mint_refresh_token = oauth.create_refresh_token

    async def revoke_after_the_claim(*args, **kwargs):
        await revoke_user_oauth_grant(world.user_id, world.app_id)
        return await mint_refresh_token(*args, **kwargs)

    monkeypatch.setattr(oauth, "create_refresh_token", revoke_after_the_claim)

    with pytest.raises(InvalidGrantError, match="revoked"):
        await refresh_tokens(grant.refresh_token, world.app_id)

    live = await OAuthAccessToken.prisma().count(
        where={"userId": world.user_id, "revokedAt": None}
    ) + await OAuthRefreshToken.prisma().count(
        where={"userId": world.user_id, "revokedAt": None}
    )
    assert live == 0
    assert await list_user_oauth_grants(world.user_id) == []


async def _sign_in(app_id: str, user_id: str, scopes: list[APIKeyPermission]) -> Grant:
    family_id = str(uuid4())
    refresh = await create_refresh_token(app_id, user_id, scopes, family_id=family_id)
    access = await create_access_token(
        app_id, user_id, scopes, refresh_family_id=family_id
    )
    return Grant(
        access_token=access.token.get_secret_value(),
        refresh_token=refresh.token.get_secret_value(),
    )


async def _age_tokens(app_id: str, user_id: str, by: timedelta) -> None:
    """Move a user's tokens for an app back in time, expiries included."""
    owned = {"userId": user_id, "applicationId": app_id}
    for model in (OAuthAccessToken, OAuthRefreshToken):
        for row in await model.prisma().find_many(where=owned):
            await model.prisma().update(
                where={"id": row.id},
                data={"createdAt": row.createdAt - by, "expiresAt": row.expiresAt - by},
            )
