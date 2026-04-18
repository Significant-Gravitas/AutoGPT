"""Shared helpers for draining and injecting pending messages.

Used by both the baseline and SDK copilot paths to avoid duplicating
the try/except drain, format, insert, and persist patterns.

Also provides the call-rate-limit check for the queue endpoint so
routes.py stays free of Redis/Lua details.
"""

import logging
from typing import TYPE_CHECKING, Callable

from backend.copilot.model import ChatMessage, upsert_chat_session
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    PendingMessageContext,
    drain_pending_messages,
    format_pending_as_user_message,
    push_pending_message,
)
from backend.copilot.stream_registry import get_session as get_active_session_meta
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import incr_with_ttl

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession
    from backend.copilot.transcript_builder import TranscriptBuilder

logger = logging.getLogger(__name__)

# Call-frequency cap for the pending-message endpoint.  The token-budget
# check guards against overspend but not rapid-fire pushes from a client
# with a large budget.
PENDING_CALL_LIMIT = 30
PENDING_CALL_WINDOW_SECONDS = 60
_PENDING_CALL_KEY_PREFIX = "copilot:pending:calls:"


async def is_turn_in_flight(session_id: str) -> bool:
    """Return ``True`` when a copilot turn is actively running for *session_id*.

    Used by the unified POST /stream entry point and the autopilot block so
    a second message arriving while an earlier turn is still executing gets
    queued into the pending buffer instead of racing the in-flight turn on
    the cluster lock.
    """
    active = await get_active_session_meta(session_id)
    return active is not None and active.status == "running"


class QueueResult:
    """Return shape of ``queue_user_message``.

    Mirrors the old ``QueuePendingMessageResponse`` so the HTTP handler can
    serialise it 1:1 and the autopilot block can surface the same fields.
    """

    __slots__ = ("buffer_length", "max_buffer_length", "turn_in_flight")

    def __init__(
        self, buffer_length: int, max_buffer_length: int, turn_in_flight: bool
    ) -> None:
        self.buffer_length = buffer_length
        self.max_buffer_length = max_buffer_length
        self.turn_in_flight = turn_in_flight


async def queue_user_message(
    *,
    session_id: str,
    message: str,
    context: PendingMessageContext | None = None,
    file_ids: list[str] | None = None,
) -> QueueResult:
    """Push *message* into the per-session pending buffer.

    The shared primitive for "a message arrived while a turn is in flight" —
    called from the unified POST /stream handler and the autopilot block.
    Call-frequency rate limiting is the caller's responsibility (HTTP path
    enforces it; internal block callers skip it).
    """
    pending = PendingMessage(
        content=message,
        file_ids=file_ids or [],
        context=context,
    )
    new_len = await push_pending_message(session_id, pending)
    return QueueResult(
        buffer_length=new_len,
        max_buffer_length=MAX_PENDING_MESSAGES,
        turn_in_flight=await is_turn_in_flight(session_id),
    )


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


async def persist_pending_as_user_rows(
    session: "ChatSession",
    transcript_builder: "TranscriptBuilder",
    pending: list[PendingMessage],
    *,
    log_prefix: str,
    content_of: Callable[[PendingMessage], str] = lambda pm: pm.content,
    on_rollback: Callable[[int], None] | None = None,
) -> bool:
    """Append ``pending`` as user rows to *session* + *transcript_builder*,
    persist, and roll back + re-queue if the persist silently failed.

    This is the shared mid-turn follow-up persist used by both the baseline
    and SDK paths — they differ only in (a) how they derive the displayed
    string from a ``PendingMessage`` and (b) what extra per-path state
    (e.g. ``openai_messages``) needs trimming on rollback.  Those variance
    points are exposed as ``content_of`` and ``on_rollback``.

    Flow:
      1. Snapshot transcript + record the session.messages length.
      2. Append one user row per pending message to both stores.
      3. ``persist_session_safe`` — swallowed errors mean no sequences get
         back-filled, which we use as the failure signal.
      4. If any newly-appended row has ``sequence is None`` → rollback:
         delete the appended rows, restore the transcript snapshot, call
         ``on_rollback(anchor)`` for the caller's own state, then re-push
         each ``PendingMessage`` into the primary pending buffer so the
         next turn-start drain picks them up.

    Returns ``True`` when the rows were persisted with sequences, ``False``
    when the rollback path fired.  Callers can use this to decide whether
    to log success or continue a retry loop.
    """
    if not pending:
        return True

    session_anchor = len(session.messages)
    transcript_snapshot = transcript_builder.snapshot()

    for pm in pending:
        content = content_of(pm)
        session.messages.append(ChatMessage(role="user", content=content))
        transcript_builder.append_user(content=content)

    session = await persist_session_safe(session, log_prefix)
    newly_appended = session.messages[session_anchor:]

    if any(m.sequence is None for m in newly_appended):
        logger.warning(
            "%s Mid-turn follow-up persist did not back-fill sequences; "
            "rolling back %d row(s) and re-queueing into the primary buffer",
            log_prefix,
            len(pending),
        )
        del session.messages[session_anchor:]
        transcript_builder.restore(transcript_snapshot)
        if on_rollback is not None:
            on_rollback(session_anchor)
        for pm in pending:
            try:
                await push_pending_message(session.session_id, pm)
            except Exception:
                logger.exception(
                    "%s Failed to re-queue mid-turn follow-up on rollback",
                    log_prefix,
                )
        return False

    logger.info(
        "%s Persisted %d mid-turn follow-up user row(s)",
        log_prefix,
        len(pending),
    )
    return True
