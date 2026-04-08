"""
Platform Bot Linking API routes.

Enables linking external chat platform servers (Discord guilds, Telegram groups,
Slack workspaces, etc.) to AutoGPT user accounts. The first user to authenticate
a server becomes the "owner" — all usage from that server is attributed to their
AutoGPT account, while each individual user gets their own CoPilot session.

Flow:
  1. Bot receives a message in an unlinked server.
  2. Bot calls POST /api/platform-linking/tokens to create a link token,
     passing the server ID and the ID of the user who triggered it.
  3. Bot DMs that user: "Set up CoPilot for this server: {frontend}/link/{token}"
  4. User clicks the link, logs in to AutoGPT, confirms → server is linked.
  5. On subsequent messages, bot calls POST /api/platform-linking/resolve.
     If linked, it calls POST /api/platform-linking/chat/stream to get a response.
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Path, Security
from prisma.models import PlatformLink, PlatformLinkToken

from .auth import check_bot_api_key, get_bot_api_key
from .models import (
    ConfirmLinkResponse,
    CreateLinkTokenRequest,
    DeleteLinkResponse,
    LinkTokenResponse,
    LinkTokenStatusResponse,
    PlatformLinkInfo,
    ResolveRequest,
    ResolveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

LINK_TOKEN_EXPIRY_MINUTES = 30

TokenPath = Annotated[
    str,
    Path(max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]


# ── Bot-facing endpoints (API key auth) ───────────────────────────────


@router.post(
    "/tokens",
    response_model=LinkTokenResponse,
    summary="Create a link token for an unlinked server",
)
async def create_link_token(
    request: CreateLinkTokenRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> LinkTokenResponse:
    """
    Called by the bot when it receives a message from an unlinked server.
    Generates a one-time token the triggering user can click to become the owner.
    """
    check_bot_api_key(x_bot_api_key)

    platform = request.platform.value

    # Reject if server is already linked
    existing = await PlatformLink.prisma().find_first(
        where={
            "platform": platform,
            "platformServerId": request.platform_server_id,
        }
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail="This server is already linked to an AutoGPT account.",
        )

    # Invalidate any pending tokens for this server
    await PlatformLinkToken.prisma().update_many(
        where={
            "platform": platform,
            "platformServerId": request.platform_server_id,
            "usedAt": None,
        },
        data={"usedAt": datetime.now(timezone.utc)},
    )

    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=LINK_TOKEN_EXPIRY_MINUTES
    )

    await PlatformLinkToken.prisma().create(
        data={
            "token": token,
            "platform": platform,
            "platformServerId": request.platform_server_id,
            "platformUserId": request.platform_user_id,
            "platformUsername": request.platform_username,
            "serverName": request.server_name,
            "channelId": request.channel_id,
            "expiresAt": expires_at,
        }
    )

    logger.info(
        "Created link token for %s server %s (expires %s)",
        platform,
        request.platform_server_id,
        expires_at.isoformat(),
    )

    link_base_url = os.getenv(
        "PLATFORM_LINK_BASE_URL", "https://platform.agpt.co/link"
    )

    return LinkTokenResponse(
        token=token,
        expires_at=expires_at,
        link_url=f"{link_base_url}/{token}",
    )


@router.get(
    "/tokens/{token}/status",
    response_model=LinkTokenStatusResponse,
    summary="Check if a link token has been consumed",
)
async def get_link_token_status(
    token: TokenPath,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> LinkTokenStatusResponse:
    """Called by the bot to check if a user has completed server linking."""
    check_bot_api_key(x_bot_api_key)

    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found.")

    if link_token.usedAt is not None:
        return LinkTokenStatusResponse(status="linked")

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return LinkTokenStatusResponse(status="expired")

    return LinkTokenStatusResponse(status="pending")


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    summary="Check whether a platform server is linked",
)
async def resolve_platform_server(
    request: ResolveRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> ResolveResponse:
    """
    Called by the bot on every incoming message to check whether the server
    has a linked AutoGPT owner account.
    """
    check_bot_api_key(x_bot_api_key)

    link = await PlatformLink.prisma().find_first(
        where={
            "platform": request.platform.value,
            "platformServerId": request.platform_server_id,
        }
    )

    return ResolveResponse(linked=link is not None)


# ── User-facing endpoints (JWT auth) ──────────────────────────────────


@router.post(
    "/tokens/{token}/confirm",
    response_model=ConfirmLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Confirm a link token (user must be authenticated)",
)
async def confirm_link_token(
    token: TokenPath,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ConfirmLinkResponse:
    """
    Called by the frontend when the user clicks the link and is logged in.
    Atomically consumes the token and creates the server → owner link.
    """
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found.")

    if link_token.usedAt is not None:
        raise HTTPException(status_code=410, detail="This link has already been used.")

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="This link has expired.")

    # Atomically mark token as used (only if still unused — prevents double-click)
    updated = await PlatformLinkToken.prisma().update_many(
        where={"token": token, "usedAt": None},
        data={"usedAt": datetime.now(timezone.utc)},
    )

    if updated == 0:
        raise HTTPException(status_code=410, detail="This link has already been used.")

    # Check if this server is already linked (race condition guard)
    existing = await PlatformLink.prisma().find_first(
        where={
            "platform": link_token.platform,
            "platformServerId": link_token.platformServerId,
        }
    )
    if existing:
        detail = (
            "This server is already linked to your account."
            if existing.userId == user_id
            else "This server is already linked to another AutoGPT account."
        )
        raise HTTPException(status_code=409, detail=detail)

    try:
        await PlatformLink.prisma().create(
            data={
                "userId": user_id,
                "platform": link_token.platform,
                "platformServerId": link_token.platformServerId,
                "ownerPlatformUserId": link_token.platformUserId,
                "serverName": link_token.serverName,
            }
        )
    except Exception as exc:
        if "unique" in str(exc).lower():
            raise HTTPException(
                status_code=409,
                detail="This server was just linked by another request.",
            ) from exc
        raise

    logger.info(
        "Linked %s server %s to user ...%s (owner: %s)",
        link_token.platform,
        link_token.platformServerId,
        user_id[-8:],
        link_token.platformUserId,
    )

    return ConfirmLinkResponse(
        success=True,
        platform=link_token.platform,
        platform_server_id=link_token.platformServerId,
        server_name=link_token.serverName,
    )


@router.get(
    "/links",
    response_model=list[PlatformLinkInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List all platform servers linked to the authenticated user",
)
async def list_my_links(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[PlatformLinkInfo]:
    """Returns all platform servers where the current user is the owner."""
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


@router.delete(
    "/links/{link_id}",
    response_model=DeleteLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a platform server",
)
async def delete_link(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DeleteLinkResponse:
    """
    Removes a platform server link. The bot will stop working in that server
    until someone links it again.
    """
    link = await PlatformLink.prisma().find_unique(where={"id": link_id})

    if not link:
        raise HTTPException(status_code=404, detail="Link not found.")

    if link.userId != user_id:
        raise HTTPException(status_code=403, detail="Not your link.")

    await PlatformLink.prisma().delete(where={"id": link_id})

    logger.info(
        "Unlinked %s server %s from user ...%s",
        link.platform,
        link.platformServerId,
        user_id[-8:],
    )

    return DeleteLinkResponse(success=True)
