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

from pydantic import BaseModel, Field, ValidationError

from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import capped_rpush, capped_rpush_if_hash_field

logger = logging.getLogger(__name__)

# Per-session cap; typing faster than the copilot drains is already unusual.
MAX_PENDING_MESSAGES = 10

# Ephemeral buffer: undrained messages are safe to drop at TTL expiry.
_PENDING_KEY_PREFIX = "copilot:pending:"
_PENDING_CHANNEL_PREFIX = "copilot:pending:notify:"
_PENDING_TTL_SECONDS = 3600  # 1 hour — matches stream_ttl default

# Secondary queue: carries drained-but-awaiting-persist PendingMessages from
# the tool wrapper (which injects them into tool output) to sdk/service.py
# (which persists a user row after the tool_result row).
_PERSIST_QUEUE_KEY_PREFIX = "copilot:pending-persist:"

# Payload sent on the pub/sub notify channel.  Subscribers treat any
# message as a wake-up hint; the value itself is not meaningful.
_NOTIFY_PAYLOAD = "1"


class PendingMessageContext(BaseModel):
    """Structured page context attached to a pending message.

    Unknown keys are silently dropped: the upstream request model is
    ``dict[str, str]``, so strict validation here only adds 500 footguns.
    """

    url: str | None = Field(default=None, max_length=2_000)
    content: str | None = Field(default=None, max_length=32_000)


class PendingMessage(BaseModel):
    """A user message queued for injection into an in-flight turn."""

    content: str = Field(min_length=1, max_length=32_000)
    file_ids: list[str] = Field(default_factory=list, max_length=20)
    context: PendingMessageContext | None = None
    # Enqueue time (unix seconds) so the turn-start drain can order pending
    # messages relative to the turn's ``current`` message.
    enqueued_at: float = Field(default_factory=time.time)


def _buffer_key(session_id: str) -> str:
    # Hash-tag braces colocate this key with stream_registry's session-meta key
    # on the same Redis Cluster slot, which the gated-rpush Lua script needs
    # (multi-key scripts return CROSSSLOT when KEYS hash to different slots).
    return f"{_PENDING_KEY_PREFIX}{{{session_id}}}"


def _notify_channel(session_id: str) -> str:
    return f"{_PENDING_CHANNEL_PREFIX}{session_id}"


def _decode_redis_item(item: Any) -> str:
    """Decode a redis-py list item to str (handles ``bytes`` and ``str``)."""
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


async def push_pending_message(
    session_id: str,
    message: PendingMessage,
) -> int:
    """Append a pending message to the session's buffer, capped at
    ``MAX_PENDING_MESSAGES`` (oldest trimmed). Returns the new buffer length.

    The buffer survives consumer crashes until ``_PENDING_TTL_SECONDS``
    expires; messages not drained within that window are dropped.
    """
    redis = await get_redis_async()
    key = _buffer_key(session_id)
    payload = message.model_dump_json()

    new_length = await capped_rpush(
        redis,
        key,
        payload,
        max_len=MAX_PENDING_MESSAGES,
        ttl_seconds=_PENDING_TTL_SECONDS,
    )

    # Fire-and-forget wake-up hint via sharded pub/sub (SPUBLISH routes to
    # one shard vs classic PUBLISH's cluster-bus broadcast). Use
    # execute_command because redis-py 6.x AsyncRedisCluster has no
    # spublish() wrapper.
    try:
        await redis.execute_command(
            "SPUBLISH", _notify_channel(session_id), _NOTIFY_PAYLOAD
        )
    except Exception as e:  # pragma: no cover
        logger.warning("pending_messages: publish failed for %s: %s", session_id, e)

    logger.info(
        "pending_messages: pushed message to session=%s (buffer_len=%d)",
        session_id,
        new_length,
    )
    return new_length


async def push_pending_message_if_session_running(
    session_id: str,
    message: PendingMessage,
    *,
    session_meta_key: str,
) -> int | None:
    """Append a pending message only while the stream meta is still running."""
    redis = await get_redis_async()
    key = _buffer_key(session_id)
    payload = message.model_dump_json()

    new_length = await capped_rpush_if_hash_field(
        redis,
        hash_key=session_meta_key,
        hash_field="status",
        expected="running",
        list_key=key,
        value=payload,
        max_len=MAX_PENDING_MESSAGES,
        ttl_seconds=_PENDING_TTL_SECONDS,
    )
    if new_length is None:
        logger.info(
            "pending_messages: skipped push to session=%s because no running turn exists",
            session_id,
        )
        return None

    # Match push_pending_message: SPUBLISH via execute_command so it works on
    # both Redis and AsyncRedisCluster (the cluster client has no publish()).
    try:
        await redis.execute_command(
            "SPUBLISH", _notify_channel(session_id), _NOTIFY_PAYLOAD
        )
    except Exception as e:  # pragma: no cover
        logger.warning("pending_messages: publish failed for %s: %s", session_id, e)

    logger.info(
        "pending_messages: pushed message to running session=%s (buffer_len=%d)",
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

    # LPOP with count drains everything in one round-trip; the push side
    # caps the list at MAX_PENDING_MESSAGES so nothing is left behind.
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


async def clear_pending_messages_unsafe(session_id: str) -> None:
    """Drop the session's pending buffer — operator/debug escape hatch.

    The ``_unsafe`` suffix warns: normal turn cleanup uses the atomic LPOP
    drain; this bypass drops queued follow-ups on the floor.
    """
    redis = await get_redis_async()
    await redis.delete(_buffer_key(session_id))


# Per-message + total caps keep the follow-up block bounded relative to the
# 100 KB MCP tool-output truncation boundary.
_FOLLOWUP_CONTENT_MAX_CHARS = 2_000
_FOLLOWUP_TOTAL_MAX_CHARS = 6_000


def _persist_queue_key(session_id: str) -> str:
    return f"{_PERSIST_QUEUE_KEY_PREFIX}{session_id}"


async def stash_pending_for_persist(
    session_id: str,
    messages: list[PendingMessage],
) -> None:
    """Enqueue drained PendingMessages for UI-row persistence.

    The SDK service LPOPs this right after appending the tool_result row so
    the user bubble lands after the tool output. Stash failures are logged
    but not raised — the only consequence is a missing UI bubble.
    """
    if not messages:
        return
    try:
        redis = await get_redis_async()
        key = _persist_queue_key(session_id)
        payloads = [m.model_dump_json() for m in messages]
        await redis.rpush(key, *payloads)  # type: ignore[misc]
        await redis.expire(key, _PENDING_TTL_SECONDS)  # type: ignore[misc]
    except Exception:
        logger.warning(
            "pending_messages: failed to stash %d message(s) for persist "
            "(session=%s); UI will miss the follow-up bubble but Claude "
            "already saw the content in tool output",
            len(messages),
            session_id,
            exc_info=True,
        )


async def drain_pending_for_persist(session_id: str) -> list[PendingMessage]:
    """Atomically drain the persist queue for *session_id*.

    Returns the queued ``PendingMessage`` objects in enqueue order (oldest
    first).  Returns ``[]`` on any error so the service-layer caller can
    always treat the result as a plain list.  Called by sdk/service.py
    after appending a tool_result row to ``session.messages``.
    """
    try:
        redis = await get_redis_async()
        key = _persist_queue_key(session_id)
        lpop_result = await redis.lpop(  # type: ignore[assignment]
            key, MAX_PENDING_MESSAGES
        )
    except Exception:
        logger.warning(
            "pending_messages: drain_pending_for_persist failed for session=%s",
            session_id,
            exc_info=True,
        )
        return []
    if not lpop_result:
        return []
    raw_popped: list[Any] = list(lpop_result)
    messages: list[PendingMessage] = []
    for item in raw_popped:
        try:
            messages.append(
                PendingMessage.model_validate(json.loads(_decode_redis_item(item)))
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            logger.warning(
                "pending_messages: dropping malformed persist-queue entry for %s: %s",
                session_id,
                e,
            )
    return messages


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
    total_chars = 0
    dropped = 0
    for idx, pm in enumerate(pending, start=1):
        text = pm.content
        if len(text) > _FOLLOWUP_CONTENT_MAX_CHARS:
            text = text[:_FOLLOWUP_CONTENT_MAX_CHARS] + "… [truncated]"
        entry = f"Message {idx}:\n{text}"
        if pm.context and pm.context.url:
            entry += f"\n[Page URL: {pm.context.url}]"
        if pm.file_ids:
            entry += "\n[Attached files: " + ", ".join(pm.file_ids) + "]"
        if total_chars + len(entry) > _FOLLOWUP_TOTAL_MAX_CHARS:
            dropped = len(pending) - idx + 1
            break
        rendered.append(entry)
        total_chars += len(entry)
    if dropped:
        rendered.append(f"… [{dropped} more message(s) truncated]")
    body = "\n\n".join(rendered)
    return (
        "<user_follow_up>\n"
        "The user sent the following message(s) while this tool was running. "
        "Treat them as a continuation of their current request — acknowledge "
        "and act on them in your next response. Do not echo these tags back.\n\n"
        f"{body}\n"
        "</user_follow_up>"
    )


async def drain_and_format_for_injection(
    session_id: str,
    *,
    log_prefix: str,
) -> str:
    """Drain the pending buffer and produce a ``<user_follow_up>`` block.

    Shared entry point for every mid-turn injection site (``PostToolUse``
    hook for MCP + built-in tools, baseline between-rounds drain, etc.).
    Also stashes the drained messages on the persist queue so the service
    layer appends a real user row after the tool_result it rode in on —
    giving the UI a correctly-ordered bubble.

    Returns an empty string if nothing was queued or Redis failed; callers
    can pass the result straight to ``additionalContext``.
    """
    if not session_id:
        return ""
    try:
        pending = await drain_pending_messages(session_id)
    except Exception:
        logger.warning(
            "%s drain_pending_messages failed (session=%s); skipping injection",
            log_prefix,
            session_id,
            exc_info=True,
        )
        return ""
    if not pending:
        return ""
    logger.info(
        "%s Injected %d user follow-up(s) into tool output (session=%s)",
        log_prefix,
        len(pending),
        session_id,
    )
    await stash_pending_for_persist(session_id, pending)
    return format_pending_as_followup(pending)


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
