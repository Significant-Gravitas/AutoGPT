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
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

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

# Payload sent on the pub/sub notify channel.  Subscribers treat any
# message as a wake-up hint; the value itself is not meaningful.
_NOTIFY_PAYLOAD = "1"


class PendingMessageContext(BaseModel, extra="forbid"):
    """Structured page context attached to a pending message."""

    url: str | None = None
    content: str | None = None


class PendingMessage(BaseModel):
    """A user message queued for injection into an in-flight turn."""

    content: str = Field(min_length=1, max_length=16_000)
    file_ids: list[str] = Field(default_factory=list)
    context: PendingMessageContext | None = None


def _buffer_key(session_id: str) -> str:
    return f"{_PENDING_KEY_PREFIX}{session_id}"


def _notify_channel(session_id: str) -> str:
    return f"{_PENDING_CHANNEL_PREFIX}{session_id}"


# Lua script: push-then-trim-then-expire-then-length, atomically.
# Redis serializes EVAL commands, so a concurrent ``LPOP`` drain
# observes either the pre-push or post-push state of the list — never
# a partial state where the RPUSH has landed but LTRIM hasn't run.
_PUSH_LUA = """
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[2]), -1)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return redis.call('LLEN', KEYS[1])
"""


async def push_pending_message(
    session_id: str,
    message: PendingMessage,
) -> int:
    """Append a pending message to the session's buffer atomically.

    Returns the new buffer length.  Enforces ``MAX_PENDING_MESSAGES`` by
    trimming from the left (oldest) — the newest message always wins if
    the user has been typing faster than the copilot can drain.

    The push + trim + expire + llen are wrapped in a single Lua EVAL so
    concurrent LPOP drains from the executor never observe a partial
    state.
    """
    redis = await get_redis_async()
    key = _buffer_key(session_id)
    payload = message.model_dump_json()

    new_length = int(
        await cast(
            "Any",
            redis.eval(
                _PUSH_LUA,
                1,
                key,
                payload,
                str(MAX_PENDING_MESSAGES),
                str(_PENDING_TTL_SECONDS),
            ),
        )
    )

    # Fire-and-forget notify.  Subscribers use this as a wake-up hint;
    # the buffer itself is authoritative so a lost notify is harmless.
    try:
        await redis.publish(_notify_channel(session_id), _NOTIFY_PAYLOAD)
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
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
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

    Not called by the normal turn flow — the atomic ``LPOP`` drain at
    turn start is the primary consumer, and any push that arrives
    after the drain window belongs to the next turn by definition.
    Retained as an operator/debug escape hatch for manually clearing a
    stuck session and as a fixture in the unit tests.
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
        if message.context.url:
            parts.append(f"\n\n[Page URL: {message.context.url}]")
        if message.context.content:
            parts.append(f"\n\n[Page content]\n{message.context.content}")
    if message.file_ids:
        parts.append(
            "\n\n[Attached files]\n"
            + "\n".join(f"- file_id={fid}" for fid in message.file_ids)
            + "\nUse read_workspace_file with the file_id to access file contents."
        )
    return {"role": "user", "content": "".join(parts)}
