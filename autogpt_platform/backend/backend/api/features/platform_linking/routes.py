"""
Platform Bot Linking API routes.

Enables linking external chat platform identities (Discord, Telegram, Slack, etc.)
to AutoGPT user accounts. Used by the multi-platform CoPilot bot.

Flow:
  1. Bot calls POST /api/platform-linking/tokens to create a link token
     for an unlinked platform user.
  2. Bot sends the user a link: {frontend}/link/{token}
  3. User clicks the link, logs in to AutoGPT, and the frontend calls
     POST /api/platform-linking/tokens/{token}/confirm to complete the link.
  4. Bot can poll GET /api/platform-linking/tokens/{token}/status or just
     check on next message via GET /api/platform-linking/resolve.
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

# Path parameter with validation for link tokens
TokenPath = Annotated[
    str,
    Path(max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]


# ── Bot-facing endpoints (API key auth) ───────────────────────────────


@router.post(
    "/tokens",
    response_model=LinkTokenResponse,
    summary="Create a link token for an unlinked platform user",
)
async def create_link_token(
    request: CreateLinkTokenRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> LinkTokenResponse:
    """
    Called by the bot service when it encounters an unlinked user.
    Generates a one-time token the user can use to link their account.
    """
    check_bot_api_key(x_bot_api_key)

    platform = request.platform.value

    # Check if already linked
    existing = await PlatformLink.prisma().find_first(
        where={
            "platform": platform,
            "platformUserId": request.platform_user_id,
        }
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail="This platform account is already linked.",
        )

    # Invalidate any existing pending tokens for this user
    await PlatformLinkToken.prisma().update_many(
        where={
            "platform": platform,
            "platformUserId": request.platform_user_id,
            "usedAt": None,
        },
        data={"usedAt": datetime.now(timezone.utc)},
    )

    # Generate token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=LINK_TOKEN_EXPIRY_MINUTES
    )

    await PlatformLinkToken.prisma().create(
        data={
            "token": token,
            "platform": platform,
            "platformUserId": request.platform_user_id,
            "platformUsername": request.platform_username,
            "channelId": request.channel_id,
            "expiresAt": expires_at,
        }
    )

    logger.info(
        "Created link token for %s (expires %s)",
        platform,
        expires_at.isoformat(),
    )

    link_base_url = os.getenv(
        "PLATFORM_LINK_BASE_URL", "https://platform.agpt.co/link"
    )
    link_url = f"{link_base_url}/{token}"

    return LinkTokenResponse(
        token=token,
        expires_at=expires_at,
        link_url=link_url,
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
    """
    Called by the bot service to check if a user has completed linking.
    """
    check_bot_api_key(x_bot_api_key)

    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found")

    if link_token.usedAt is not None:
        # Token was used — find the linked account
        link = await PlatformLink.prisma().find_first(
            where={
                "platform": link_token.platform,
                "platformUserId": link_token.platformUserId,
            }
        )
        return LinkTokenStatusResponse(
            status="linked",
            user_id=link.userId if link else None,
        )

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return LinkTokenStatusResponse(status="expired")

    return LinkTokenStatusResponse(status="pending")


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    summary="Resolve a platform identity to an AutoGPT user",
)
async def resolve_platform_user(
    request: ResolveRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> ResolveResponse:
    """
    Called by the bot service on every incoming message to check if
    the platform user has a linked AutoGPT account.
    """
    check_bot_api_key(x_bot_api_key)

    link = await PlatformLink.prisma().find_first(
        where={
            "platform": request.platform.value,
            "platformUserId": request.platform_user_id,
        }
    )

    if not link:
        return ResolveResponse(linked=False)

    return ResolveResponse(linked=True, user_id=link.userId)


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
    Consumes the token and creates the platform link.
    Uses atomic update_many to prevent race conditions on double-click.
    """
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found.")

    if link_token.usedAt is not None:
        raise HTTPException(status_code=410, detail="This link has already been used.")

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="This link has expired.")

    # Atomically mark token as used (only if still unused)
    updated = await PlatformLinkToken.prisma().update_many(
        where={"token": token, "usedAt": None},
        data={"usedAt": datetime.now(timezone.utc)},
    )

    if updated == 0:
        raise HTTPException(status_code=410, detail="This link has already been used.")

    # Check if this platform identity is already linked
    existing = await PlatformLink.prisma().find_first(
        where={
            "platform": link_token.platform,
            "platformUserId": link_token.platformUserId,
        }
    )
    if existing:
        detail = (
            "This platform account is already linked to your account."
            if existing.userId == user_id
            else "This platform account is already linked to another user."
        )
        raise HTTPException(status_code=409, detail=detail)

    # Create the link — catch unique constraint race condition
    try:
        await PlatformLink.prisma().create(
            data={
                "userId": user_id,
                "platform": link_token.platform,
                "platformUserId": link_token.platformUserId,
                "platformUsername": link_token.platformUsername,
            }
        )
    except Exception as exc:
        if "unique" in str(exc).lower():
            raise HTTPException(
                status_code=409,
                detail="This platform account was just linked by another request.",
            ) from exc
        raise

    logger.info(
        "Linked %s:%s to user ...%s",
        link_token.platform,
        link_token.platformUserId,
        user_id[-8:],
    )

    return ConfirmLinkResponse(
        success=True,
        platform=link_token.platform,
        platform_user_id=link_token.platformUserId,
        platform_username=link_token.platformUsername,
    )


@router.get(
    "/links",
    response_model=list[PlatformLinkInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List all platform links for the authenticated user",
)
async def list_my_links(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[PlatformLinkInfo]:
    """Returns all platform identities linked to the current user's account."""
    links = await PlatformLink.prisma().find_many(
        where={"userId": user_id},
        order={"linkedAt": "desc"},
    )

    return [
        PlatformLinkInfo(
            id=link.id,
            platform=link.platform,
            platform_user_id=link.platformUserId,
            platform_username=link.platformUsername,
            linked_at=link.linkedAt,
        )
        for link in links
    ]


@router.delete(
    "/links/{link_id}",
    response_model=DeleteLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a platform identity",
)
async def delete_link(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DeleteLinkResponse:
    """
    Removes a platform link. The user will need to re-link if they
    want to use the bot on that platform again.
    """
    link = await PlatformLink.prisma().find_unique(where={"id": link_id})

    if not link:
        raise HTTPException(status_code=404, detail="Link not found.")

    if link.userId != user_id:
        raise HTTPException(status_code=403, detail="Not your link.")

    await PlatformLink.prisma().delete(where={"id": link_id})

    logger.info(
        "Unlinked %s:%s from user ...%s",
        link.platform,
        link.platformUserId,
        user_id[-8:],
    )

    return DeleteLinkResponse(success=True)
