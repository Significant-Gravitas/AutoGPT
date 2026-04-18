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

    url: str | None = Field(default=None, max_length=2_000)
    content: str | None = Field(default=None, max_length=32_000)


class PendingMessage(BaseModel):
    """A user message queued for injection into an in-flight turn."""

    content: str = Field(min_length=1, max_length=32_000)
    file_ids: list[str] = Field(default_factory=list, max_length=20)
    context: PendingMessageContext | None = None


def _buffer_key(session_id: str) -> str:
    return f"{_PENDING_KEY_PREFIX}{session_id}"


def _notify_channel(session_id: str) -> str:
    return f"{_PENDING_CHANNEL_PREFIX}{session_id}"


def _decode_redis_item(item: Any) -> str:
    """Decode a redis-py list item to a str.

    redis-py returns ``bytes`` when ``decode_responses=False`` and ``str``
    when ``decode_responses=True``.  This helper handles both so callers
    don't have to repeat the isinstance guard.
    """
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


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
    """Append a pending message to the session's buffer.

    Returns the new buffer length.  Enforces ``MAX_PENDING_MESSAGES`` by
    trimming from the left (oldest) — the newest message always wins if
    the user has been typing faster than the copilot can drain.

    Executed as a single Lua EVAL so RPUSH + LTRIM + EXPIRE + LLEN run
    atomically in one round-trip; a concurrent drain (LPOP) can no longer
    observe the list temporarily over ``MAX_PENDING_MESSAGES``.

    Note on durability: if the executor turn crashes after a push but before
    the drain window runs, the message remains in Redis until the TTL expires
    (``_PENDING_TTL_SECONDS``, currently 1 hour).  It is delivered on the
    next turn that drains the buffer.  If no turn runs within the TTL the
    message is silently dropped; the user may resend it.
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
    # Draining MAX_PENDING_MESSAGES at once is safe because the push side
    # uses RPUSH + LTRIM(-MAX_PENDING_MESSAGES, -1) to cap the list to that
    # same value, so the list can never hold more items than we drain here.
    # If the cap is raised on the push side, raise the drain count here too
    # (or switch to a loop drain).
    lpop_result = await redis.lpop(key, MAX_PENDING_MESSAGES)  # type: ignore[assignment]
    if not lpop_result:
        return []
    raw_popped: list[Any] = list(lpop_result)

    # redis-py may return bytes or str depending on decode_responses.
    decoded: list[str] = [_decode_redis_item(item) for item in raw_popped]

    messages: list[PendingMessage] = []
    for payload in decoded:
        try:
            messages.append(PendingMessage.model_validate(json.loads(payload)))
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


async def peek_pending_messages(session_id: str) -> list[PendingMessage]:
    """Return pending messages without consuming them.

    Uses LRANGE 0 -1 to read all items in enqueue order (oldest first)
    without removing them.  Returns an empty list if the buffer is empty
    or the session has no pending messages.
    """
    redis = await get_redis_async()
    key = _buffer_key(session_id)
    items = await cast("Any", redis.lrange(key, 0, -1))
    if not items:
        return []
    messages: list[PendingMessage] = []
    for item in items:
        try:
            messages.append(
                PendingMessage.model_validate(json.loads(_decode_redis_item(item)))
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            logger.warning(
                "pending_messages: dropping malformed peek entry for %s: %s",
                session_id,
                e,
            )
    return messages


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


# Per-message cap for inline tool-boundary injection.  Keeps the follow-up
# block from crowding out the actual tool output on large inputs.  Queued
# messages longer than this are truncated with an ellipsis marker.
_FOLLOWUP_CONTENT_MAX_CHARS = 2_000


def format_pending_as_followup(pending: list[PendingMessage]) -> str:
    """Render drained pending messages as a ``<user_follow_up>`` block.

    Used by the SDK tool-boundary injection path to surface queued user
    text inside a tool result so the model reads it on the next LLM round,
    without starting a separate turn.  Wrapped in a stable XML-style tag so
    the shared system-prompt supplement can teach the model to treat the
    contents as the user's continuation of their request, not as tool
    output.  Each message is capped to keep the block bounded even if the
    user pastes long content.
    """
    if not pending:
        return ""
    rendered: list[str] = []
    for idx, pm in enumerate(pending, start=1):
        text = pm.content
        if len(text) > _FOLLOWUP_CONTENT_MAX_CHARS:
            text = text[:_FOLLOWUP_CONTENT_MAX_CHARS] + "… [truncated]"
        rendered.append(f"Message {idx}:\n{text}")
        if pm.context and pm.context.url:
            rendered[-1] += f"\n[Page URL: {pm.context.url}]"
        if pm.file_ids:
            rendered[-1] += "\n[Attached files: " + ", ".join(pm.file_ids) + "]"
    body = "\n\n".join(rendered)
    return (
        "<user_follow_up>\n"
        "The user sent the following message(s) while this tool was running. "
        "Treat them as a continuation of their current request — acknowledge "
        "and act on them in your next response. Do not echo these tags back.\n\n"
        f"{body}\n"
        "</user_follow_up>"
    )


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
