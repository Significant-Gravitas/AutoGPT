"""The OAuth apps a user has given access to their account, as that user sees them."""

from datetime import datetime, timezone
from typing import Optional

from prisma.enums import APIKeyPermission
from prisma.models import OAuthAccessToken as PrismaOAuthAccessToken
from prisma.models import OAuthAuthorizationCode as PrismaOAuthAuthorizationCode
from prisma.models import OAuthRefreshToken as PrismaOAuthRefreshToken
from pydantic import BaseModel, Field


class OAuthGrantInfo(BaseModel):
    """An app that holds a live access or refresh token for the user."""

    application_id: str
    name: str
    logo_url: Optional[str] = None
    scopes: list[APIKeyPermission]
    last_active_at: datetime = Field(description="When the app last got a token")


async def list_user_oauth_grants(user_id: str) -> list[OAuthGrantInfo]:
    """The apps holding a live token for the user, most recently active first."""
    now = datetime.now(timezone.utc)
    refresh_tokens = await PrismaOAuthRefreshToken.prisma().find_many(
        where={
            "userId": user_id,
            "revokedAt": None,
            "expiresAt": {"gt": now},
        },
        include={"Application": True},
    )
    access_tokens = await PrismaOAuthAccessToken.prisma().find_many(
        where={"userId": user_id, "revokedAt": None, "expiresAt": {"gt": now}},
        include={"Application": True},
    )

    grants: dict[str, OAuthGrantInfo] = {}
    for token in [*refresh_tokens, *access_tokens]:
        if not token.Application:
            continue
        grant = grants.get(token.applicationId)
        if grant is None:
            grant = grants[token.applicationId] = OAuthGrantInfo(
                application_id=token.applicationId,
                name=token.Application.name,
                logo_url=token.Application.logoUrl,
                scopes=[],
                last_active_at=token.createdAt,
            )
        grant.scopes = _ordered({*grant.scopes, *token.scopes})
        grant.last_active_at = max(grant.last_active_at, token.createdAt)

    return sorted(grants.values(), key=lambda g: g.last_active_at, reverse=True)


async def revoke_user_oauth_grant(user_id: str, application_id: str) -> int:
    """End every token and authorization code one app holds for one user.

    Returns how many it ended, so 0 means the app held none for this user,
    whoever's app it is.
    """
    now = datetime.now(timezone.utc)

    # Every use of a token checks its family, so this also ends a pair minted after it.
    await PrismaOAuthRefreshToken.prisma().update_many(
        where={
            "userId": user_id,
            "applicationId": application_id,
            "familyRevokedAt": None,
        },
        data={"familyRevokedAt": now},
    )
    codes = await PrismaOAuthAuthorizationCode.prisma().delete_many(
        where={"userId": user_id, "applicationId": application_id}
    )
    refresh = await PrismaOAuthRefreshToken.prisma().update_many(
        where={"userId": user_id, "applicationId": application_id, "revokedAt": None},
        data={"revokedAt": now},
    )
    access = await PrismaOAuthAccessToken.prisma().update_many(
        where={"userId": user_id, "applicationId": application_id, "revokedAt": None},
        data={"revokedAt": now},
    )
    return codes + refresh + access


def _ordered(scopes: set[APIKeyPermission]) -> list[APIKeyPermission]:
    return [p for p in APIKeyPermission if p in scopes]
