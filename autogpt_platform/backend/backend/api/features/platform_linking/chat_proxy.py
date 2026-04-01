"""
Bot Chat Proxy endpoints.

Allows the bot service to send messages to CoPilot on behalf of
linked users, authenticated via bot API key.
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


@router.post(
    "/chat/session",
    response_model=BotChatSessionResponse,
    summary="Create a CoPilot session for a linked user (bot-facing)",
)
async def bot_create_session(
    request: BotChatRequest,
    x_bot_api_key: str | None = Depends(get_bot_api_key),
) -> BotChatSessionResponse:
    """Creates a new CoPilot chat session on behalf of a linked user."""
    check_bot_api_key(x_bot_api_key)

    link = await PlatformLink.prisma().find_first(where={"userId": request.user_id})
    if not link:
        raise HTTPException(status_code=404, detail="User has no platform links.")

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
    check_bot_api_key(x_bot_api_key)

    user_id = request.user_id

    # Verify user has a platform link
    link = await PlatformLink.prisma().find_first(where={"userId": user_id})
    if not link:
        raise HTTPException(status_code=404, detail="User has no platform links.")

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
