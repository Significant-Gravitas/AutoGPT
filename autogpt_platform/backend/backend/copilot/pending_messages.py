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
from backend.data.redis_helpers import capped_rpush

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

# Secondary queue that carries drained-but-awaiting-persist PendingMessages
# from the MCP tool wrapper (which drains the primary buffer and injects
# into tool output for the LLM) to sdk/service.py's _dispatch_response
# handler for StreamToolOutputAvailable, which pops and persists them as a
# separate user row chronologically after the tool_result row.  This is the
# hand-off between "Claude saw the follow-up mid-turn" (wrapper) and "UI
# renders a user bubble for it" (service).  Rollback path re-queues into
# the PRIMARY buffer so the next turn-start drain picks them up if the
# user-row persist fails.
_PERSIST_QUEUE_KEY_PREFIX = "copilot:pending-persist:"

# Payload sent on the pub/sub notify channel.  Subscribers treat any
# message as a wake-up hint; the value itself is not meaningful.
_NOTIFY_PAYLOAD = "1"


class PendingMessageContext(BaseModel):
    """Structured page context attached to a pending message.

    Default ``extra='ignore'`` (pydantic's default): unknown keys from
    the loose HTTP-level ``StreamChatRequest.context: dict[str, str]``
    are silently dropped rather than raising ``ValidationError`` on
    forward-compat additions.  The strict ``extra='forbid'`` mode was
    removed after sentry r3105553772 — strict validation at this
    boundary only added a 500 footgun; the upstream request model is
    already schemaless so strict mode protects nothing.
    """

    url: str | None = Field(default=None, max_length=2_000)
    content: str | None = Field(default=None, max_length=32_000)


class PendingMessage(BaseModel):
    """A user message queued for injection into an in-flight turn."""

    content: str = Field(min_length=1, max_length=32_000)
    file_ids: list[str] = Field(default_factory=list, max_length=20)
    context: PendingMessageContext | None = None
    # Wall-clock time (unix seconds, float) the message was queued by the
    # user.  Used by the turn-start drain to order pending relative to the
    # turn's ``current`` message: items typed *before* the current's
    # /stream arrival go ahead of it; items typed *after* (race path,
    # queued while the /stream HTTP request was still processing) go
    # after.  Defaults to 0.0 for backward compatibility with entries
    # written before this field existed — those sort as "before everything"
    # which matches the pre-fix behaviour.
    enqueued_at: float = Field(default_factory=time.time)


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


async def push_pending_message(
    session_id: str,
    message: PendingMessage,
) -> int:
    """Append a pending message to the session's buffer.

    Returns the new buffer length.  Enforces ``MAX_PENDING_MESSAGES`` by
    trimming from the left (oldest) — the newest message always wins if
    the user has been typing faster than the copilot can drain.

    Delegates to :func:`backend.data.redis_helpers.capped_rpush` so RPUSH
    + LTRIM + EXPIRE + LLEN run atomically (MULTI/EXEC) in one round
    trip; a concurrent drain (LPOP) can no longer observe the list
    temporarily over ``MAX_PENDING_MESSAGES``.

    Note on durability: if the executor turn crashes after a push but before
    the drain window runs, the message remains in Redis until the TTL expires
    (``_PENDING_TTL_SECONDS``, currently 1 hour).  It is delivered on the
    next turn that drains the buffer.  If no turn runs within the TTL the
    message is silently dropped; the user may resend it.
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


async def _clear_pending_messages_unsafe(session_id: str) -> None:
    """Drop the session's pending buffer — **not** the normal turn cleanup.

    Named ``_unsafe`` because reaching for this at turn end drops queued
    follow-ups on the floor instead of running them (the bug fixed by
    commit b64be73).  The atomic ``LPOP`` drain at turn start is the
    primary consumer; anything pushed after the drain window belongs to
    the next turn by definition.  Retained only as an operator/debug
    escape hatch for manually clearing a stuck session and as a fixture
    in the unit tests.
    """
    redis = await get_redis_async()
    await redis.delete(_buffer_key(session_id))


# Per-message and total-block caps for inline tool-boundary injection.
# Per-message keeps a single long paste from dominating; the total cap
# keeps the follow-up block small relative to the 100 KB MCP truncation
# boundary so tool output always stays the larger share of the wrapper
# return value.
_FOLLOWUP_CONTENT_MAX_CHARS = 2_000
_FOLLOWUP_TOTAL_MAX_CHARS = 6_000


def _persist_queue_key(session_id: str) -> str:
    return f"{_PERSIST_QUEUE_KEY_PREFIX}{session_id}"


async def stash_pending_for_persist(
    session_id: str,
    messages: list[PendingMessage],
) -> None:
    """Enqueue drained PendingMessages for UI-row persistence.

    Writes each message as a JSON payload to
    ``copilot:pending-persist:{session_id}``.  The SDK service's
    tool-result dispatch handler LPOPs this queue right after appending
    the tool_result row to ``session.messages``, so the resulting user
    row lands at the correct chronological position (after the tool
    output the follow-up was drained against).

    Fire-and-forget on Redis failures: a stash failure means Claude
    still saw the follow-up in tool output (the injection step ran
    first), so the only consequence is a missing UI bubble.  Logged
    so it can be spotted.
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
                "pending_messages: dropping malformed persist-queue entry "
                "for %s: %s",
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
