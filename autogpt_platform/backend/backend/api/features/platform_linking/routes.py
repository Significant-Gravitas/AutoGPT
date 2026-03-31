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
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field

import backend.data.db

logger = logging.getLogger(__name__)

router = APIRouter()

LINK_TOKEN_EXPIRY_MINUTES = 30


# ── Request / Response Models ──────────────────────────────────────────


class CreateLinkTokenRequest(BaseModel):
    """Request from the bot service to create a linking token."""

    platform: str = Field(
        description="Platform name: DISCORD, TELEGRAM, SLACK, TEAMS, WHATSAPP, GITHUB, LINEAR"
    )
    platform_user_id: str = Field(description="The user's ID on the platform")
    platform_username: str | None = Field(
        default=None, description="Display name (best effort)"
    )
    channel_id: str | None = Field(
        default=None, description="Channel ID for sending confirmation back"
    )


class LinkTokenResponse(BaseModel):
    token: str
    expires_at: datetime
    link_url: str


class LinkTokenStatusResponse(BaseModel):
    status: str  # "pending", "linked", "expired"
    user_id: str | None = None


class ResolveRequest(BaseModel):
    """Resolve a platform identity to an AutoGPT user."""

    platform: str
    platform_user_id: str


class ResolveResponse(BaseModel):
    linked: bool
    user_id: str | None = None
    platform_username: str | None = None


class PlatformLinkInfo(BaseModel):
    id: str
    platform: str
    platform_user_id: str
    platform_username: str | None
    linked_at: datetime


class ConfirmLinkResponse(BaseModel):
    success: bool
    platform: str
    platform_user_id: str
    platform_username: str | None


# ── Bot-facing endpoints (API key auth) ───────────────────────────────


@router.post(
    "/tokens",
    response_model=LinkTokenResponse,
    summary="Create a link token for an unlinked platform user",
)
async def create_link_token(
    request: CreateLinkTokenRequest,
) -> LinkTokenResponse:
    """
    Called by the bot service when it encounters an unlinked user.
    Generates a one-time token the user can use to link their account.

    TODO: Add API key auth for bot service (for now, open for development).
    """
    platform = request.platform.upper()
    _validate_platform(platform)

    prisma = backend.data.db.get_prisma()

    # Check if already linked
    existing = await prisma.platformlink.find_unique(
        where={
            "platform_platformUserId": {
                "platform": platform,
                "platformUserId": request.platform_user_id,
            }
        }
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Platform user {request.platform_user_id} on {platform} "
            f"is already linked to an account.",
        )

    # Generate token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=LINK_TOKEN_EXPIRY_MINUTES
    )

    await prisma.platformlinktoken.create(
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
        f"Created link token for {platform}:{request.platform_user_id} "
        f"(expires {expires_at.isoformat()})"
    )

    # TODO: Make base URL configurable
    link_url = f"https://platform.agpt.co/link/{token}"

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
async def get_link_token_status(token: str) -> LinkTokenStatusResponse:
    """
    Called by the bot service to check if a user has completed linking.
    """
    prisma = backend.data.db.get_prisma()

    link_token = await prisma.platformlinktoken.find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found")

    if link_token.usedAt is not None:
        # Token was used — find the linked account
        link = await prisma.platformlink.find_unique(
            where={
                "platform_platformUserId": {
                    "platform": link_token.platform,
                    "platformUserId": link_token.platformUserId,
                }
            }
        )
        return LinkTokenStatusResponse(
            status="linked",
            user_id=link.userId if link else None,
        )

    if link_token.expiresAt < datetime.now(timezone.utc):
        return LinkTokenStatusResponse(status="expired")

    return LinkTokenStatusResponse(status="pending")


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    summary="Resolve a platform identity to an AutoGPT user",
)
async def resolve_platform_user(
    request: ResolveRequest,
) -> ResolveResponse:
    """
    Called by the bot service on every incoming message to check if
    the platform user has a linked AutoGPT account.
    """
    platform = request.platform.upper()
    _validate_platform(platform)

    prisma = backend.data.db.get_prisma()

    link = await prisma.platformlink.find_unique(
        where={
            "platform_platformUserId": {
                "platform": platform,
                "platformUserId": request.platform_user_id,
            }
        }
    )

    if not link:
        return ResolveResponse(linked=False)

    return ResolveResponse(
        linked=True,
        user_id=link.userId,
        platform_username=link.platformUsername,
    )


# ── User-facing endpoints (JWT auth) ──────────────────────────────────


@router.post(
    "/tokens/{token}/confirm",
    response_model=ConfirmLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Confirm a link token (user must be authenticated)",
)
async def confirm_link_token(
    token: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ConfirmLinkResponse:
    """
    Called by the frontend when the user clicks the link and is logged in.
    Consumes the token and creates the platform link.
    """
    prisma = backend.data.db.get_prisma()

    link_token = await prisma.platformlinktoken.find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found")

    if link_token.usedAt is not None:
        raise HTTPException(status_code=410, detail="Token already used")

    if link_token.expiresAt < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="Token expired")

    # Check if this platform identity is already linked to someone else
    existing = await prisma.platformlink.find_unique(
        where={
            "platform_platformUserId": {
                "platform": link_token.platform,
                "platformUserId": link_token.platformUserId,
            }
        }
    )
    if existing:
        if existing.userId == user_id:
            raise HTTPException(
                status_code=409,
                detail="This platform account is already linked to your account.",
            )
        raise HTTPException(
            status_code=409,
            detail="This platform account is already linked to another user.",
        )

    # Create the link
    await prisma.platformlink.create(
        data={
            "userId": user_id,
            "platform": link_token.platform,
            "platformUserId": link_token.platformUserId,
            "platformUsername": link_token.platformUsername,
        }
    )

    # Mark token as used
    await prisma.platformlinktoken.update(
        where={"token": token},
        data={"usedAt": datetime.now(timezone.utc)},
    )

    logger.info(
        f"Linked {link_token.platform}:{link_token.platformUserId} "
        f"to user {user_id[-8:]}"
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
    """
    Returns all platform identities linked to the current user's account.
    """
    prisma = backend.data.db.get_prisma()

    links = await prisma.platformlink.find_many(
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
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a platform identity",
)
async def delete_link(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> dict:
    """
    Removes a platform link. The user will need to re-link if they
    want to use the bot on that platform again.
    """
    prisma = backend.data.db.get_prisma()

    link = await prisma.platformlink.find_unique(where={"id": link_id})

    if not link:
        raise HTTPException(status_code=404, detail="Link not found")

    if link.userId != user_id:
        raise HTTPException(status_code=403, detail="Not your link")

    await prisma.platformlink.delete(where={"id": link_id})

    logger.info(
        f"Unlinked {link.platform}:{link.platformUserId} from user {user_id[-8:]}"
    )

    return {"success": True}


# ── Helpers ────────────────────────────────────────────────────────────

VALID_PLATFORMS = {
    "DISCORD",
    "TELEGRAM",
    "SLACK",
    "TEAMS",
    "WHATSAPP",
    "GITHUB",
    "LINEAR",
}


def _validate_platform(platform: str) -> None:
    if platform not in VALID_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform '{platform}'. Must be one of: {', '.join(sorted(VALID_PLATFORMS))}",
        )
