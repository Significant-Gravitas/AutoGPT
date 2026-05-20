"""Chat-turn orchestration for the platform bot bridge."""

import logging
from uuid import uuid4

from backend.copilot import stream_registry
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.model import (
    ChatMessage,
    append_and_save_message,
    create_chat_session,
    get_chat_session,
)
from backend.data.db_accessors import platform_linking_db
from backend.util.exceptions import DuplicateChatMessageError, NotFoundError

from .models import BotChatRequest, ChatTurnHandle

logger = logging.getLogger(__name__)

CHAT_TOOL_CALL_ID = "chat_stream"
CHAT_TOOL_NAME = "chat"


async def resolve_chat_owner(request: BotChatRequest) -> str:
    """Return the AutoGPT user ID that owns the platform conversation.

    Server context → server owner. DM context → the DM-linked user.
    """
    platform = request.platform.value
    db = platform_linking_db()

    if request.platform_server_id:
        owner = await db.find_server_link_owner(platform, request.platform_server_id)
        if owner is None:
            raise NotFoundError("This server is not linked to an AutoGPT account.")
        return owner

    owner = await db.find_user_link_owner(platform, request.platform_user_id)
    if owner is None:
        raise NotFoundError("Your DMs are not linked to an AutoGPT account.")
    return owner


async def start_chat_turn(request: BotChatRequest) -> ChatTurnHandle:
    """Prepare a copilot turn; caller subscribes via the returned handle.

    ``subscribe_from="0-0"`` on the handle means a late subscriber replays
    the full stream (Redis Streams, not pub/sub).
    """
    owner_user_id = await resolve_chat_owner(request)

    session_id = request.session_id
    if session_id:
        session = await get_chat_session(session_id, owner_user_id)
        if not session:
            raise NotFoundError("Session not found.")
    else:
        session = await create_chat_session(owner_user_id, dry_run=False)
        session_id = session.session_id

    # Persist the user message before enqueueing, mirroring the REST chat
    # endpoint — otherwise the executor runs against empty history.
    is_duplicate = (
        await append_and_save_message(
            session_id, ChatMessage(role="user", content=request.message)
        )
    ) is None
    if is_duplicate:
        # Matches REST chat behaviour: skip create_session + enqueue so we
        # don't create an orphan stream with no producer. Caller subscribes
        # to the in-flight turn via its own retry logic, or drops.
        logger.info(
            "Duplicate bot message for session %s (platform %s, user ...%s)",
            session_id,
            request.platform.value,
            owner_user_id[-8:],
        )
        raise DuplicateChatMessageError("Message already in flight.")

    turn_id = str(uuid4())

    await stream_registry.create_session(
        session_id=session_id,
        user_id=owner_user_id,
        tool_call_id=CHAT_TOOL_CALL_ID,
        tool_name=CHAT_TOOL_NAME,
        turn_id=turn_id,
    )

    await enqueue_copilot_turn(
        session_id=session_id,
        user_id=owner_user_id,
        message=request.message,
        turn_id=turn_id,
        is_user_message=True,
    )

    logger.info(
        "Bot chat turn started: %s (server %s, session %s, turn %s, owner ...%s)",
        request.platform.value,
        request.platform_server_id or "DM",
        session_id,
        turn_id,
        owner_user_id[-8:],
    )

    return ChatTurnHandle(
        session_id=session_id,
        turn_id=turn_id,
        user_id=owner_user_id,
    )
