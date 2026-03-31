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

import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Literal

from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from prisma.models import PlatformLink, PlatformLinkToken
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

LINK_TOKEN_EXPIRY_MINUTES = 30
LINK_BASE_URL = os.getenv("PLATFORM_LINK_BASE_URL", "https://platform.agpt.co/link")


# ── Platform enum (mirrors Prisma PlatformType) ───────────────────────


class Platform(str, Enum):
    DISCORD = "DISCORD"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    WHATSAPP = "WHATSAPP"
    GITHUB = "GITHUB"
    LINEAR = "LINEAR"


# ── API Key auth for bot-facing endpoints ─────────────────────────────

BOT_API_KEY = os.getenv("PLATFORM_BOT_API_KEY", "")


async def get_bot_api_key(request: Request) -> str | None:
    """Extract the bot API key from the X-Bot-API-Key header."""
    return request.headers.get("x-bot-api-key")


def _check_bot_api_key(api_key: str | None) -> None:
    """Validate the bot API key. Uses constant-time comparison."""
    if not BOT_API_KEY:
        # No key configured — allow in development only
        if os.getenv("ENV", "development") != "development":
            raise HTTPException(
                status_code=503,
                detail="Bot API key not configured.",
            )
        return
    if not api_key or not hmac.compare_digest(api_key, BOT_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid bot API key.")


# ── Request / Response Models ──────────────────────────────────────────


class CreateLinkTokenRequest(BaseModel):
    """Request from the bot service to create a linking token."""

    platform: Platform = Field(description="Platform name")
    platform_user_id: str = Field(
        description="The user's ID on the platform",
        min_length=1,
        max_length=255,
    )
    platform_username: str | None = Field(
        default=None,
        description="Display name (best effort)",
        max_length=255,
    )
    channel_id: str | None = Field(
        default=None,
        description="Channel ID for sending confirmation back",
        max_length=255,
    )


class LinkTokenResponse(BaseModel):
    token: str
    expires_at: datetime
    link_url: str


class LinkTokenStatusResponse(BaseModel):
    status: Literal["pending", "linked", "expired"]
    user_id: str | None = None


class ResolveRequest(BaseModel):
    """Resolve a platform identity to an AutoGPT user."""

    platform: Platform
    platform_user_id: str = Field(min_length=1, max_length=255)


class ResolveResponse(BaseModel):
    linked: bool
    user_id: str | None = None


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


class DeleteLinkResponse(BaseModel):
    success: bool


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
    _check_bot_api_key(x_bot_api_key)

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

    link_url = f"{LINK_BASE_URL}/{token}"

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
    token: str,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> LinkTokenStatusResponse:
    """
    Called by the bot service to check if a user has completed linking.
    """
    _check_bot_api_key(x_bot_api_key)

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
    _check_bot_api_key(x_bot_api_key)

    link = await PlatformLink.prisma().find_first(
        where={
            "platform": request.platform.value,
            "platformUserId": request.platform_user_id,
        }
    )

    if not link:
        return ResolveResponse(linked=False)

    return ResolveResponse(
        linked=True,
        user_id=link.userId,
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
    Uses atomic update_many to prevent race conditions on double-click.
    """
    # Fetch and validate token
    link_token = await PlatformLinkToken.prisma().find_unique(where={"token": token})

    if not link_token:
        raise HTTPException(status_code=404, detail="Token not found.")

    if link_token.usedAt is not None:
        raise HTTPException(status_code=410, detail="This link has already been used.")

    if link_token.expiresAt.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="This link has expired.")

    # Atomically mark token as used (only if still unused)
    updated = await PlatformLinkToken.prisma().update_many(
        where={
            "token": token,
            "usedAt": None,
        },
        data={"usedAt": datetime.now(timezone.utc)},
    )

    if updated == 0:
        # Another request already consumed it
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
        # Handle race condition: another request linked this identity
        if "unique" in str(exc).lower() or "Unique" in str(exc):
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
    """
    Returns all platform identities linked to the current user's account.
    """
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


# ── Bot Chat Proxy ────────────────────────────────────────────────────
# Allows the bot service to send messages to CoPilot on behalf of
# linked users, authenticated via bot API key.


class BotChatRequest(BaseModel):
    """Request from the bot to chat as a linked user."""

    user_id: str = Field(description="The linked AutoGPT user ID")
    message: str = Field(
        description="The user's message", min_length=1, max_length=32000
    )
    session_id: str | None = Field(
        default=None,
        description="Existing chat session ID. If omitted, a new session is created.",
    )


class BotChatSessionResponse(BaseModel):
    """Returned when creating a new session via the bot proxy."""

    session_id: str


@router.post(
    "/chat/session",
    response_model=BotChatSessionResponse,
    summary="Create a CoPilot session for a linked user (bot-facing)",
)
async def bot_create_session(
    request: BotChatRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> BotChatSessionResponse:
    """
    Creates a new CoPilot chat session on behalf of a linked user.
    """
    _check_bot_api_key(x_bot_api_key)

    # Verify this user has a platform link (bot shouldn't call for unlinked users)
    link = await PlatformLink.prisma().find_first(where={"userId": request.user_id})
    if not link:
        raise HTTPException(
            status_code=404,
            detail="User has no platform links.",
        )

    from backend.copilot.model import create_chat_session

    session = await create_chat_session(request.user_id)

    return BotChatSessionResponse(session_id=session.session_id)


@router.post(
    "/chat/stream",
    summary="Stream a CoPilot response for a linked user (bot-facing)",
)
async def bot_chat_stream(
    request: BotChatRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
):
    """
    Send a message to CoPilot on behalf of a linked user and stream
    the response back as Server-Sent Events.

    The bot authenticates with its API key — no user JWT needed.
    """
    import asyncio
    from uuid import uuid4

    from fastapi.responses import StreamingResponse

    from backend.copilot import stream_registry
    from backend.copilot.executor.utils import enqueue_copilot_turn
    from backend.copilot.model import (
        ChatMessage,
        append_and_save_message,
        create_chat_session,
        get_chat_session,
    )
    from backend.copilot.response_model import StreamFinish

    _check_bot_api_key(x_bot_api_key)

    user_id = request.user_id

    # Verify user has a platform link
    link = await PlatformLink.prisma().find_first(where={"userId": user_id})
    if not link:
        raise HTTPException(
            status_code=404,
            detail="User has no platform links.",
        )

    # Get or create session
    session_id = request.session_id
    if session_id:
        session = await get_chat_session(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        session = await create_chat_session(user_id)
        session_id = session.session_id

    # Save user message
    message = ChatMessage(role="user", content=request.message)
    await append_and_save_message(session_id, message)

    # Create a turn and enqueue
    turn_id = str(uuid4())

    await stream_registry.create_session(
        session_id=session_id,
        user_id=user_id,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id=turn_id,
    )

    subscribe_from_id = "0-0"

    await enqueue_copilot_turn(
        session_id=session_id,
        user_id=user_id,
        message=request.message,
        turn_id=turn_id,
        is_user_message=True,
    )

    logger.info(
        "Bot chat: user ...%s, session %s, turn %s",
        user_id[-8:],
        session_id,
        turn_id,
    )

    # Stream SSE response
    async def event_generator():
        subscriber_queue = None
        try:
            subscriber_queue = await stream_registry.subscribe_to_session(
                session_id=session_id,
                user_id=user_id,
                last_message_id=subscribe_from_id,
            )

            if subscriber_queue is None:
                yield StreamFinish().to_sse()
                yield "data: [DONE]\n\n"
                return

            while True:
                try:
                    chunk = await asyncio.wait_for(subscriber_queue.get(), timeout=30.0)

                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield chunk.to_sse()

                    if isinstance(chunk, StreamFinish) or (
                        isinstance(chunk, str) and "[DONE]" in chunk
                    ):
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"

        except Exception:
            logger.exception("Bot chat stream error for session %s", session_id)
            yield 'data: {"type": "error", "content": "Stream error"}\n\n'
            yield "data: [DONE]\n\n"
        finally:
            if subscriber_queue is not None:
                await stream_registry.unsubscribe_from_session(
                    session_id=session_id,
                    subscriber_queue=subscriber_queue,
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )
