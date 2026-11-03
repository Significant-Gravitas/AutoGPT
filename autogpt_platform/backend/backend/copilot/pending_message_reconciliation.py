import logging
from collections.abc import Callable

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.data.db_accessors import chat_db

logger = logging.getLogger(__name__)


async def reconcile_pending_rows(
    session: ChatSession,
    pending: list[PendingMessage],
    content_of: Callable[[PendingMessage], str],
) -> list[ChatMessage]:
    unique: dict[str, PendingMessage] = {}
    for message in pending:
        unique.setdefault(message.id, message)
    try:
        stored = await chat_db().get_chat_messages_by_ids(
            session.session_id, list(unique)
        )
    except Exception:
        logger.exception("Failed to reconcile pending IDs; persistence will retry")
        stored = []
    by_id = {message.id: message for message in stored}
    return [
        by_id.get(message.id)
        or ChatMessage(
            id=message.id,
            role="user",
            content=content_of(message),
            metadata=message.metadata or None,
        )
        for message in unique.values()
    ]
