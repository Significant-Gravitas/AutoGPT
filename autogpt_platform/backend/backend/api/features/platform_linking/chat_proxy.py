"""
Bot Chat Proxy endpoints.

Allows the bot service to send messages to CoPilot on behalf of a linked
server's users, authenticated via bot API key.

The bot never handles AutoGPT user IDs — it passes platform_server_id and
platform_user_id, and the backend resolves the owner internally. This
prevents impersonation even if the bot API key is compromised.

Each (platform_server_id, platform_user_id) pair gets its own CoPilot
session, all owned by the server owner's AutoGPT account. The owner can see
all conversations in their AutoGPT account.
"""

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from prisma.models import PlatformLink

from backend.copilot import stream_registry
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.model import (
    ChatMessage,
    append_and_save_message,
    create_chat_session,
    get_chat_session,
)
from backend.copilot.response_model import StreamFinish

from .auth import check_bot_api_key, get_bot_api_key
from .models import BotChatRequest, BotChatSessionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _resolve_owner(platform: str, platform_server_id: str) -> str:
    """
    Look up the AutoGPT owner user ID for a linked server.
    Raises 404 if the server is not linked.
    """
    link = await PlatformLink.prisma().find_first(
        where={"platform": platform, "platformServerId": platform_server_id}
    )
    if not link:
        raise HTTPException(
            status_code=404,
            detail="This server is not linked to an AutoGPT account.",
        )
    return link.userId


@router.post(
    "/chat/session",
    response_model=BotChatSessionResponse,
    summary="Create a CoPilot session for a server user (bot-facing)",
)
async def bot_create_session(
    request: BotChatRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> BotChatSessionResponse:
    """
    Creates a new CoPilot session on behalf of a platform user in a linked server.
    The session is owned by the server owner's AutoGPT account.
    """
    check_bot_api_key(x_bot_api_key)

    owner_user_id = await _resolve_owner(
        request.platform.value, request.platform_server_id
    )
    session = await create_chat_session(owner_user_id)

    logger.info(
        "Bot created session %s for %s user %s in server %s (owner ...%s)",
        session.session_id,
        request.platform.value,
        request.platform_user_id,
        request.platform_server_id,
        owner_user_id[-8:],
    )

    return BotChatSessionResponse(session_id=session.session_id)


@router.post(
    "/chat/stream",
    summary="Stream a CoPilot response for a server user (bot-facing)",
)
async def bot_chat_stream(
    request: BotChatRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
):
    """
    Send a message to CoPilot on behalf of a platform user in a linked server,
    streaming the response back as Server-Sent Events.

    The bot authenticates with its API key — no user JWT required.
    The owner's AutoGPT account is resolved from the server link.
    """
    check_bot_api_key(x_bot_api_key)

    owner_user_id = await _resolve_owner(
        request.platform.value, request.platform_server_id
    )

    # Get or create session
    session_id = request.session_id
    if session_id:
        session = await get_chat_session(session_id, owner_user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        session = await create_chat_session(owner_user_id)
        session_id = session.session_id

    message = ChatMessage(role="user", content=request.message)
    await append_and_save_message(session_id, message)

    turn_id = str(uuid4())

    await stream_registry.create_session(
        session_id=session_id,
        user_id=owner_user_id,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id=turn_id,
    )

    subscribe_from_id = "0-0"

    await enqueue_copilot_turn(
        session_id=session_id,
        user_id=owner_user_id,
        message=request.message,
        turn_id=turn_id,
        is_user_message=True,
    )

    logger.info(
        "Bot chat: %s server %s, user %s, session %s, turn %s (owner ...%s)",
        request.platform.value,
        request.platform_server_id,
        request.platform_user_id,
        session_id,
        turn_id,
        owner_user_id[-8:],
    )

    async def event_generator():
        subscriber_queue = None
        try:
            subscriber_queue = await stream_registry.subscribe_to_session(
                session_id=session_id,
                user_id=owner_user_id,
                last_message_id=subscribe_from_id,
            )

            if subscriber_queue is None:
                yield StreamFinish().to_sse()
                yield "data: [DONE]\n\n"
                return

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        subscriber_queue.get(), timeout=30.0
                    )

                    yield chunk if isinstance(chunk, str) else chunk.to_sse()

                    if isinstance(chunk, StreamFinish) or (
                        isinstance(chunk, str) and "[DONE]" in chunk
                    ):
                        break

                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

        except Exception:
            logger.exception(
                "Bot chat stream error for session %s", session_id
            )
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
