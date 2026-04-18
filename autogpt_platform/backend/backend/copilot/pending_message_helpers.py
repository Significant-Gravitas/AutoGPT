"""Shared helpers for draining and injecting pending messages.

Used by both the baseline and SDK copilot paths to avoid duplicating
the try/except drain, format, insert, and persist patterns.

Also provides the call-rate-limit check for the queue endpoint so
routes.py stays free of Redis/Lua details.
"""

import logging
from typing import TYPE_CHECKING

from backend.copilot.model import ChatMessage, upsert_chat_session
from backend.copilot.pending_messages import (
    drain_pending_messages,
    format_pending_as_user_message,
)
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import incr_with_ttl

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession

logger = logging.getLogger(__name__)

# Call-frequency cap for the pending-message endpoint.  The token-budget
# check guards against overspend but not rapid-fire pushes from a client
# with a large budget.
PENDING_CALL_LIMIT = 30
PENDING_CALL_WINDOW_SECONDS = 60
_PENDING_CALL_KEY_PREFIX = "copilot:pending:calls:"


async def check_pending_call_rate(user_id: str) -> int:
    """Increment and return the per-user push counter for the current window.

    The counter is **user-global**: it counts pushes across ALL sessions
    belonging to the user, not per-session.  This prevents a client from
    bypassing the cap by spreading rapid pushes across many sessions.

    Returns the new call count.  Raises nothing — callers compare the
    return value against ``PENDING_CALL_LIMIT`` and decide what to do.
    Fails open (returns 0) if Redis is unavailable so the endpoint stays
    usable during Redis hiccups.
    """
    try:
        redis = await get_redis_async()
        key = f"{_PENDING_CALL_KEY_PREFIX}{user_id}"
        return await incr_with_ttl(redis, key, PENDING_CALL_WINDOW_SECONDS)
    except Exception:
        logger.warning(
            "pending_message_helpers: call-rate check failed for user=%s, failing open",
            user_id,
        )
        return 0


async def drain_pending_safe(session_id: str, log_prefix: str = "") -> list[str]:
    """Drain the pending buffer and return formatted content strings.

    Combines drain + format into one call.  Returns ``[]`` on any error
    so callers can always treat the result as a plain list.
    """
    try:
        pending = await drain_pending_messages(session_id)
    except Exception:
        logger.warning(
            "%s drain_pending_messages failed, skipping",
            log_prefix or "pending_messages",
            exc_info=True,
        )
        return []
    return [format_pending_as_user_message(pm)["content"] for pm in pending]


def insert_pending_before_last(session: "ChatSession", texts: list[str]) -> None:
    """Insert pending messages into *session* just before the last message.

    Pending messages were queued during the previous turn, so they belong
    chronologically before the current user message that was already
    appended via ``maybe_append_user_message``.  Inserting at ``len-1``
    preserves that order: [...history, pending_1, pending_2, current_msg].

    The caller must have already appended the current user message before
    calling this function.  If ``session.messages`` is unexpectedly empty,
    a warning is logged and the messages are appended at index 0 so they
    are not silently lost.
    """
    if not texts:
        return
    if not session.messages:
        logger.warning(
            "insert_pending_before_last: session.messages is empty — "
            "current user message was not appended before drain; "
            "inserting pending messages at index 0"
        )
    insert_idx = max(0, len(session.messages) - 1)
    for i, content in enumerate(texts):
        session.messages.insert(
            insert_idx + i, ChatMessage(role="user", content=content)
        )


async def persist_session_safe(
    session: "ChatSession", log_prefix: str = ""
) -> "ChatSession":
    """Persist *session* to the DB, returning the (possibly updated) session.

    Swallows transient DB errors so a failing persist doesn't discard
    messages already popped from Redis — the turn continues from memory.
    """
    try:
        return await upsert_chat_session(session)
    except Exception as err:
        logger.warning(
            "%s Failed to persist pending messages: %s",
            log_prefix or "pending_messages",
            err,
        )
        return session
