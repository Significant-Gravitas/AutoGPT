"""Platform link DB operations.

Directly accessed by the ``AgentServer`` / ``DatabaseManager`` pods (which
hold the Prisma connection). Other services go through
``backend.data.db_accessors.platform_linking_db`` so calls are transparently
routed via ``DatabaseManagerAsyncClient`` when no local Prisma is available.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone

from prisma.errors import UniqueViolationError
from prisma.models import PlatformLink, PlatformLinkToken, PlatformUserLink

from backend.data.db import transaction
from backend.util.exceptions import (
    LinkAlreadyExistsError,
    LinkFlowMismatchError,
    LinkTokenExpiredError,
    NotAuthorizedError,
    NotFoundError,
)
from backend.util.settings import Settings

from .models import (
    ConfirmLinkResponse,
    ConfirmUserLinkResponse,
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    DeleteLinkResponse,
    LinkTokenInfoResponse,
    LinkTokenResponse,
    LinkTokenStatusResponse,
    LinkType,
    PlatformLinkInfo,
    PlatformUserLinkInfo,
    ResolveResponse,
)

logger = logging.getLogger(__name__)

LINK_TOKEN_EXPIRY_MINUTES = 30


def _link_base_url() -> str:
    return Settings().config.platform_link_base_url


# ── Owner lookups ─────────────────────────────────────────────────────
# These return the owning AutoGPT user_id (or None). Using scalars instead
# of Prisma models keeps everything RPC-safe — Prisma objects are rejected
# by AppService's result validator.


async def find_server_link_owner(platform: str, platform_server_id: str) -> str | None:
    link = await PlatformLink.prisma().find_first(
        where={"platform": platform, "platformServerId": platform_server_id}
    )
    return link.userId if link else None


async def find_user_link_owner(platform: str, platform_user_id: str) -> str | None:
    link = await PlatformUserLink.prisma().find_unique(
        where={
            "platform_platformUserId": {
                "platform": platform,
                "platformUserId": platform_user_id,
            }
        }
    )
    return link.userId if link else None


async def resolve_server_link(
    platform: str, platform_server_id: str
) -> ResolveResponse:
    owner = await find_server_link_owner(platform, platform_server_id)
    return ResolveResponse(linked=owner is not None)


async def resolve_user_link(platform: str, platform_user_id: str) -> ResolveResponse:
    owner = await find_user_link_owner(platform, platform_user_id)
    return ResolveResponse(linked=owner is not None)


# ── Token creation ────────────────────────────────────────────────────


async def create_server_link_token(
    request: CreateLinkTokenRequest,
) -> LinkTokenResponse:
    platform = request.platform.value

    if await find_server_link_owner(platform, request.platform_server_id):
        raise LinkAlreadyExistsError(
            "This server is already linked to an AutoGPT account."
        )

    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=LINK_TOKEN_EXPIRY_MINUTES
    )

    # Atomic: invalidate pending tokens + create the new one, so two racing
    # create calls can't leave two valid tokens for the same target.
    async with transaction() as tx:
        await PlatformLinkToken.prisma(tx).update_many(
            where={
                "platform": platform,
                "linkType": LinkType.SERVER.value,
                "platformServerId": request.platform_server_id,
                "usedAt": None,
            },
            data={"usedAt": datetime.now(timezone.utc)},
        )
        await PlatformLinkToken.prisma(tx).create(
            data={
                "token": token,
                "platform": platform,
                "linkType": LinkType.SERVER.value,
                "platformServerId": request.platform_server_id,
                "platformUserId": request.platform_user_id,
                "platformUsername": request.platform_username,
                "serverName": request.server_name,
                "channelId": request.channel_id,
                "expiresAt": expires_at,
            }
        )

    logger.info(
        "Created SERVER link token for %s server %s (expires %s)",
        platform,
        request.platform_server_id,
        expires_at.isoformat(),
    )

    return LinkTokenResponse(
        token=token,
        expires_at=expires_at,
        link_url=f"{_link_base_url()}/{token}?platform={platform}",
    )


async def create_user_link_token(
    request: CreateUserLinkTokenRequest,
) -> LinkTokenResponse:
    platform = request.platform.value

    if await find_user_link_owner(platform, request.platform_user_id):
        raise LinkAlreadyExistsError(
            "Your DMs with the bot are already linked to an AutoGPT account."
        )

    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=LINK_TOKEN_EXPIRY_MINUTES
    )

    async with transaction() as tx:
        await PlatformLinkToken.prisma(tx).update_many(
            where={
                "platform": platform,
                "linkType": LinkType.USER.value,
                "platformUserId": request.platform_user_id,
                "usedAt": None,
            },
            data={"usedAt": datetime.now(timezone.utc)},
        )
        await PlatformLinkToken.prisma(tx).create(
            data={
                "token": token,
                "platform": platform,
                "linkType": LinkType.USER.value,
                "platformUserId": request.platform_user_id,
                "platformUsername": request.platform_username,
                "expiresAt": expires_at,
            }
        )

    logger.info(
        "Created USER link token for %s (expires %s)", platform, expires_at.isoformat()
    )

    return LinkTokenResponse(
        token=token,
        expires_at=expires_at,
        link_url=f"{_link_base_url()}/{token}?platform={platform}",
    )


# ── Token status / info ───────────────────────────────────────────────


async def get_link_token_status(token: str) -> LinkTokenStatusResponse:
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise NotFoundError("Token not found.")

    if link_token.usedAt is not None:
        # A superseded token (invalidated by create_*_token) has usedAt set
        # without a backing link row — report expired, not linked.
        if link_token.linkType == LinkType.USER.value:
            owner = await find_user_link_owner(
                link_token.platform, link_token.platformUserId
            )
        else:
            owner = (
                await find_server_link_owner(
                    link_token.platform, link_token.platformServerId
                )
                if link_token.platformServerId
                else None
            )
        return LinkTokenStatusResponse(status="linked" if owner else "expired")

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return LinkTokenStatusResponse(status="expired")

    return LinkTokenStatusResponse(status="pending")


async def get_link_token_info(token: str) -> LinkTokenInfoResponse:
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token or link_token.usedAt is not None:
        raise NotFoundError("Token not found.")

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise LinkTokenExpiredError("Token expired.")

    return LinkTokenInfoResponse(
        platform=link_token.platform,
        link_type=LinkType(link_token.linkType),
        server_name=link_token.serverName,
    )


# ── Confirmation (user-facing, JWT-authed) ────────────────────────────


async def confirm_server_link(token: str, user_id: str) -> ConfirmLinkResponse:
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise NotFoundError("Token not found.")
    if link_token.linkType != LinkType.SERVER.value:
        raise LinkFlowMismatchError("This link is for a different linking flow.")
    if link_token.usedAt is not None:
        raise LinkTokenExpiredError("This link has already been used.")
    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise LinkTokenExpiredError("This link has expired.")
    if not link_token.platformServerId:
        raise LinkFlowMismatchError("Server token missing server ID.")

    owner = await find_server_link_owner(
        link_token.platform, link_token.platformServerId
    )
    if owner:
        detail = (
            "This server is already linked to your account."
            if owner == user_id
            else "This server is already linked to another AutoGPT account."
        )
        raise LinkAlreadyExistsError(detail)

    # Atomic consume + create so a failed create doesn't burn the token.
    now = datetime.now(timezone.utc)
    try:
        async with transaction() as tx:
            updated = await PlatformLinkToken.prisma(tx).update_many(
                where={"token": token, "usedAt": None, "expiresAt": {"gt": now}},
                data={"usedAt": now},
            )
            if updated == 0:
                raise LinkTokenExpiredError("This link has already been used.")
            await PlatformLink.prisma(tx).create(
                data={
                    "userId": user_id,
                    "platform": link_token.platform,
                    "platformServerId": link_token.platformServerId,
                    "ownerPlatformUserId": link_token.platformUserId,
                    "serverName": link_token.serverName,
                }
            )
    except UniqueViolationError as exc:
        raise LinkAlreadyExistsError(
            "This server was just linked by another request."
        ) from exc

    logger.info(
        "Linked %s server %s to user ...%s",
        link_token.platform,
        link_token.platformServerId,
        user_id[-8:],
    )

    return ConfirmLinkResponse(
        success=True,
        platform=link_token.platform,
        platform_server_id=link_token.platformServerId,
        server_name=link_token.serverName,
    )


async def confirm_user_link(token: str, user_id: str) -> ConfirmUserLinkResponse:
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise NotFoundError("Token not found.")
    if link_token.linkType != LinkType.USER.value:
        raise LinkFlowMismatchError("This link is for a different linking flow.")
    if link_token.usedAt is not None:
        raise LinkTokenExpiredError("This link has already been used.")
    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise LinkTokenExpiredError("This link has expired.")

    owner = await find_user_link_owner(link_token.platform, link_token.platformUserId)
    if owner:
        detail = (
            "Your DMs are already linked to your account."
            if owner == user_id
            else "This platform user is already linked to another AutoGPT account."
        )
        raise LinkAlreadyExistsError(detail)

    now = datetime.now(timezone.utc)
    try:
        async with transaction() as tx:
            updated = await PlatformLinkToken.prisma(tx).update_many(
                where={"token": token, "usedAt": None, "expiresAt": {"gt": now}},
                data={"usedAt": now},
            )
            if updated == 0:
                raise LinkTokenExpiredError("This link has already been used.")
            await PlatformUserLink.prisma(tx).create(
                data={
                    "userId": user_id,
                    "platform": link_token.platform,
                    "platformUserId": link_token.platformUserId,
                    "platformUsername": link_token.platformUsername,
                }
            )
    except UniqueViolationError as exc:
        raise LinkAlreadyExistsError(
            "Your DMs were just linked by another request."
        ) from exc

    logger.info(
        "Linked %s DMs to AutoGPT user ...%s", link_token.platform, user_id[-8:]
    )

    return ConfirmUserLinkResponse(
        success=True,
        platform=link_token.platform,
        platform_user_id=link_token.platformUserId,
    )


# ── Listing ───────────────────────────────────────────────────────────


async def list_server_links(user_id: str) -> list[PlatformLinkInfo]:
    links = await PlatformLink.prisma().find_many(
        where={"userId": user_id},
        order={"linkedAt": "desc"},
    )
    return [
        PlatformLinkInfo(
            id=link.id,
            platform=link.platform,
            platform_server_id=link.platformServerId,
            owner_platform_user_id=link.ownerPlatformUserId,
            server_name=link.serverName,
            linked_at=link.linkedAt,
        )
        for link in links
    ]


async def list_user_links(user_id: str) -> list[PlatformUserLinkInfo]:
    links = await PlatformUserLink.prisma().find_many(
        where={"userId": user_id},
        order={"linkedAt": "desc"},
    )
    return [
        PlatformUserLinkInfo(
            id=link.id,
            platform=link.platform,
            platform_user_id=link.platformUserId,
            platform_username=link.platformUsername,
            linked_at=link.linkedAt,
        )
        for link in links
    ]


# ── Deletion ──────────────────────────────────────────────────────────


async def delete_server_link(link_id: str, user_id: str) -> DeleteLinkResponse:
    link = await PlatformLink.prisma().find_unique(where={"id": link_id})
    if not link:
        raise NotFoundError("Link not found.")
    if link.userId != user_id:
        raise NotAuthorizedError("Not your link.")

    await PlatformLink.prisma().delete(where={"id": link_id})
    logger.info(
        "Unlinked %s server %s from user ...%s",
        link.platform,
        link.platformServerId,
        user_id[-8:],
    )
    return DeleteLinkResponse(success=True)


async def delete_user_link(link_id: str, user_id: str) -> DeleteLinkResponse:
    link = await PlatformUserLink.prisma().find_unique(where={"id": link_id})
    if not link:
        raise NotFoundError("Link not found.")
    if link.userId != user_id:
        raise NotAuthorizedError("Not your link.")

    await PlatformUserLink.prisma().delete(where={"id": link_id})
    logger.info("Unlinked %s DMs from AutoGPT user ...%s", link.platform, user_id[-8:])
    return DeleteLinkResponse(success=True)


# ── Cleanup ──────────────────────────────────────────────────────────

# Keep recently-expired rows for debugging.
LINK_TOKEN_RETENTION_HOURS = 24


async def cleanup_expired_platform_link_tokens() -> int:
    """Delete PlatformLinkToken rows expired beyond the retention window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LINK_TOKEN_RETENTION_HOURS)
    deleted = await PlatformLinkToken.prisma().delete_many(
        where={"expiresAt": {"lt": cutoff}}
    )
    if deleted > 0:
        logger.info("Cleaned up %d expired platform link tokens", deleted)
    return deleted
