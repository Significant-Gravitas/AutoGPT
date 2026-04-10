"""Pending-message buffer for in-flight copilot turns.

When a user sends a new message while a copilot turn is already executing,
instead of blocking the frontend (or queueing a brand-new turn after the
current one finishes), we want the new message to be *injected into the
running turn* — appended between tool-call rounds so the model sees it
before its next LLM call.

This module provides the cross-process buffer that makes that possible:

- **Producer** (chat API route): pushes a pending message to Redis and
  publishes a notification on a pub/sub channel.
- **Consumer** (executor running the turn): on each tool-call round,
  drains the buffer and appends the pending messages to the conversation.

The Redis list is the durable store; the pub/sub channel is a fast
wake-up hint for long-idle consumers (not used by default, but available
for future blocking-wait semantics).

A hard cap of ``MAX_PENDING_MESSAGES`` per session prevents abuse.  The
buffer is trimmed to the latest ``MAX_PENDING_MESSAGES`` on every push.
"""

import json
import logging
import time
from typing import Any, cast

from pydantic import BaseModel, Field

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Per-session cap.  Higher values risk a runaway consumer; lower values
# risk dropping user input under heavy typing.  10 was chosen as a
# reasonable ceiling — a user typing faster than the copilot can drain
# between tool rounds is already an unusual usage pattern.
MAX_PENDING_MESSAGES = 10

# Redis key + TTL.  The buffer is ephemeral: if a turn completes or the
# executor dies, the pending messages should either have been drained
# already or are safe to drop (the user can resend).
_PENDING_KEY_PREFIX = "copilot:pending:"
_PENDING_CHANNEL_PREFIX = "copilot:pending:notify:"
_PENDING_TTL_SECONDS = 3600  # 1 hour — matches stream_ttl default


class PendingMessage(BaseModel):
    """A user message queued for injection into an in-flight turn."""

    content: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list)
    context: dict[str, str] | None = None
    # Unix epoch seconds at enqueue time, for ordering and debugging.
    enqueued_at: float = Field(default_factory=time.time)


def _buffer_key(session_id: str) -> str:
    return f"{_PENDING_KEY_PREFIX}{session_id}"


def _notify_channel(session_id: str) -> str:
    return f"{_PENDING_CHANNEL_PREFIX}{session_id}"


async def push_pending_message(
    session_id: str,
    message: PendingMessage,
) -> int:
    """Append a pending message to the session's buffer.

    Returns the new buffer length.  Enforces ``MAX_PENDING_MESSAGES`` by
    trimming from the left (oldest) — the newest message always wins if
    the user has been typing faster than the copilot can drain.
    """
    redis = await get_redis_async()
    key = _buffer_key(session_id)
    payload = message.model_dump_json()

    # Push + trim + expire in a pipeline so the three writes land atomically
    # enough for this use case (pipelining doesn't guarantee atomicity
    # across commands but ordering is preserved).
    async with redis.pipeline(transaction=False) as pipe:
        pipe.rpush(key, payload)
        pipe.ltrim(key, -MAX_PENDING_MESSAGES, -1)
        pipe.expire(key, _PENDING_TTL_SECONDS)
        pipe.llen(key)
        results = await pipe.execute()

    new_length = int(results[-1])

    # Fire-and-forget notify.  Subscribers use this as a wake-up hint;
    # the buffer itself is authoritative so a lost notify is harmless.
    try:
        await redis.publish(_notify_channel(session_id), "1")
    except Exception as e:  # pragma: no cover
        logger.warning("pending_messages: publish failed for %s: %s", session_id, e)

    logger.info(
        "pending_messages: pushed message to session=%s (buffer_len=%d)",
        session_id,
        new_length,
    )
    return new_length


async def drain_pending_messages(session_id: str) -> list[PendingMessage]:
    """Atomically pop all pending messages for *session_id*.

    Returns them in enqueue order (oldest first).  Uses ``LPOP`` with a
    count so the read+delete is a single Redis round trip.  If the list
    is empty or missing, returns ``[]``.
    """
    redis = await get_redis_async()
    key = _buffer_key(session_id)

    # Redis LPOP with count (Redis 6.2+) returns None for missing key,
    # empty list if we somehow race an empty key, or the popped items.
    # redis-py's async lpop overload with a count collapses the return
    # type in pyright; cast the awaitable so strict type-check stays
    # clean without changing runtime behaviour.
    lpop_result = await cast(
        "Any",
        redis.lpop(key, MAX_PENDING_MESSAGES),
    )
    if not lpop_result:
        return []
    raw_popped: list[Any] = list(lpop_result)

    # redis-py may return bytes or str depending on decode_responses.
    decoded: list[str] = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw_popped
    ]

    messages: list[PendingMessage] = []
    for payload in decoded:
        try:
            messages.append(PendingMessage(**json.loads(payload)))
        except Exception as e:
            logger.warning(
                "pending_messages: dropping malformed entry for %s: %s",
                session_id,
                e,
            )

    if messages:
        logger.info(
            "pending_messages: drained %d messages for session=%s",
            len(messages),
            session_id,
        )
    return messages


async def peek_pending_count(session_id: str) -> int:
    """Return the current buffer length without consuming it."""
    redis = await get_redis_async()
    length = await cast("Any", redis.llen(_buffer_key(session_id)))
    return int(length)


async def clear_pending_messages(session_id: str) -> None:
    """Drop the session's pending buffer.

    Called at the end of a turn (success or failure) so messages from a
    previous turn don't leak into the next one.  The buffer may already
    have been drained inside the turn — this is a safety net.
    """
    redis = await get_redis_async()
    await redis.delete(_buffer_key(session_id))


def format_pending_as_user_message(message: PendingMessage) -> dict[str, Any]:
    """Shape a ``PendingMessage`` into the OpenAI-format user message dict.

    Used by the baseline tool-call loop when injecting the buffered
    message into the conversation.  Context/file metadata (if any) is
    embedded into the content so the model sees everything in one block.
    """
    parts: list[str] = [message.content]
    if message.context:
        url = message.context.get("url")
        if url:
            parts.append(f"\n\n[Page URL: {url}]")
        page_content = message.context.get("content")
        if page_content:
            parts.append(f"\n\n[Page content]\n{page_content}")
    if message.file_ids:
        parts.append(
            "\n\n[Attached files]\n"
            + "\n".join(f"- file_id={fid}" for fid in message.file_ids)
            + "\nUse read_workspace_file with the file_id to access file contents."
        )
    return {"role": "user", "content": "".join(parts)}
