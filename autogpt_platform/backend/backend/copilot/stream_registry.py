"""Stream registry for managing reconnectable SSE streams.

This module provides a registry for tracking active streaming sessions and their
messages. It uses Redis for all state management (no in-memory state), making
pods stateless and horizontally scalable.

Architecture:
- Redis Stream: Persists all messages for replay and real-time delivery
- Redis Hash: Session metadata (status, session_id, etc.)

Subscribers:
1. Replay missed messages from Redis Stream (XREAD)
2. Listen for live updates via blocking XREAD
3. No in-memory state required on the subscribing pod
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import orjson
from redis.exceptions import RedisError

from backend.api.model import CopilotCompletionPayload
from backend.data.db_accessors import chat_db
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import hash_compare_and_set

from .config import ChatConfig
from .executor.utils import COPILOT_CONSUMER_TIMEOUT_SECONDS
from .response_model import (
    ResponseType,
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamHeartbeat,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)

logger = logging.getLogger(__name__)
config = ChatConfig()
_notification_bus = AsyncRedisNotificationEventBus()

# Track background tasks for this pod (just the asyncio.Task reference, not subscribers)
_local_sessions: dict[str, asyncio.Task] = {}

# Track listener tasks per subscriber queue for cleanup
# Maps queue id() to (session_id, asyncio.Task) for proper cleanup on unsubscribe
_listener_sessions: dict[int, tuple[str, asyncio.Task]] = {}

# Timeout for putting chunks into subscriber queues (seconds)
# If the queue is full and doesn't drain within this time, send an overflow error
QUEUE_PUT_TIMEOUT = 5.0


@dataclass
class ActiveSession:
    """Represents an active streaming session (metadata only, no in-memory queues)."""

    session_id: str
    user_id: str | None
    tool_call_id: str
    tool_name: str
    turn_id: str = ""
    blocking: bool = False  # If True, HTTP request is waiting for completion
    status: Literal["running", "completed", "failed"] = "running"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    asyncio_task: asyncio.Task | None = None


def _get_session_meta_key(session_id: str) -> str:
    """Get Redis key for session metadata (keyed by session_id)."""
    return f"{config.session_meta_prefix}{session_id}"


def _get_turn_stream_key(turn_id: str) -> str:
    """Get Redis key for turn message stream (keyed by turn_id for per-turn isolation)."""
    return f"{config.turn_stream_prefix}{turn_id}"


def _parse_session_meta(meta: dict[Any, Any], session_id: str = "") -> ActiveSession:
    """Parse a raw Redis hash into a typed ActiveSession.

    Centralises the ``meta.get(...)`` boilerplate so callers don't repeat it.
    ``session_id`` is used as a fallback for ``turn_id`` when the meta hash
    pre-dates the turn_id field (backward compat for in-flight sessions).
    """
    created_at = datetime.now(timezone.utc)
    created_at_raw = meta.get("created_at")
    if created_at_raw:
        try:
            created_at = datetime.fromisoformat(str(created_at_raw))
        except (ValueError, TypeError):
            pass

    return ActiveSession(
        session_id=meta.get("session_id", "") or session_id,
        user_id=meta.get("user_id", "") or None,
        tool_call_id=meta.get("tool_call_id", ""),
        tool_name=meta.get("tool_name", ""),
        turn_id=meta.get("turn_id", "") or session_id,
        blocking=meta.get("blocking") == "1",
        status=meta.get("status", "running"),  # type: ignore[arg-type]
        created_at=created_at,
    )


async def create_session(
    session_id: str,
    user_id: str | None,
    tool_call_id: str,
    tool_name: str,
    turn_id: str = "",
    blocking: bool = False,
) -> ActiveSession:
    """Create a new streaming session in Redis (keyed by session_id).

    Args:
        session_id: Chat session ID (used as session identifier)
        user_id: User ID (may be None for anonymous)
        tool_call_id: Tool call ID from the LLM
        tool_name: Name of the tool being executed
        turn_id: Unique per-turn UUID for stream isolation
        blocking: If True, HTTP request is waiting for completion

    Returns:
        The created ActiveSession instance (metadata only)
    """
    start_time = time.perf_counter()

    # Build log metadata for structured logging
    log_meta = {
        "component": "StreamRegistry",
        "session_id": session_id,
    }
    if user_id:
        log_meta["user_id"] = user_id

    logger.info(
        f"[TIMING] create_session STARTED, session={session_id}, user={user_id}, turn_id={turn_id}",
        extra={"json_fields": log_meta},
    )

    # Create session
    session = ActiveSession(
        session_id=session_id,
        user_id=user_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        turn_id=turn_id,
        blocking=blocking,
    )

    # Store metadata in Redis
    redis_start = time.perf_counter()
    redis = await get_redis_async()
    redis_time = (time.perf_counter() - redis_start) * 1000
    logger.info(
        f"[TIMING] get_redis_async took {redis_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": redis_time}},
    )

    meta_key = _get_session_meta_key(session_id)
    # No need to delete old stream — each turn_id is a fresh UUID

    hset_start = time.perf_counter()
    await redis.hset(  # type: ignore[misc]
        meta_key,
        mapping={
            "session_id": session_id,
            "user_id": user_id or "",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "turn_id": turn_id,
            "blocking": "1" if blocking else "0",
            "status": session.status,
            "created_at": session.created_at.isoformat(),
        },
    )
    hset_time = (time.perf_counter() - hset_start) * 1000
    logger.info(
        f"[TIMING] redis.hset took {hset_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": hset_time}},
    )

    await redis.expire(meta_key, config.stream_ttl)

    total_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"[TIMING] create_session COMPLETED in {total_time:.1f}ms; session={session_id}",
        extra={"json_fields": {**log_meta, "total_time_ms": total_time}},
    )

    return session


_meta_ttl_refresh_at: dict[str, float] = {}
"""Tracks the last time the session meta key TTL was refreshed.

Used by `publish_chunk` to avoid refreshing on every single chunk
(expensive). Refreshes at most once every 60 seconds per session.
"""

_META_TTL_REFRESH_INTERVAL = 60  # seconds


async def publish_chunk(
    turn_id: str,
    chunk: StreamBaseResponse,
    *,
    session_id: str | None = None,
) -> str:
    """Publish a chunk to Redis Stream.

    All delivery is via Redis Streams - no in-memory state.

    Args:
        turn_id: Turn ID (per-turn UUID) identifying the stream
        chunk: The stream response chunk to publish
        session_id: Chat session ID — when provided, the session meta key
            TTL is refreshed periodically to prevent expiration during
            long-running turns (see SECRT-2178).

    Returns:
        The Redis Stream message ID
    """
    start_time = time.perf_counter()
    chunk_type = type(chunk).__name__
    chunk_json = chunk.model_dump_json()
    message_id = "0-0"

    # Build log metadata
    log_meta = {
        "component": "StreamRegistry",
        "turn_id": turn_id,
        "chunk_type": chunk_type,
    }

    try:
        redis = await get_redis_async()
        stream_key = _get_turn_stream_key(turn_id)

        # Write to Redis Stream for persistence and real-time delivery
        xadd_start = time.perf_counter()
        raw_id = await redis.xadd(
            stream_key,
            {"data": chunk_json},
            maxlen=config.stream_max_length,
        )
        xadd_time = (time.perf_counter() - xadd_start) * 1000
        message_id = raw_id if isinstance(raw_id, str) else raw_id.decode()

        # Set TTL on stream to match session metadata TTL
        await redis.expire(stream_key, config.stream_ttl)

        # Periodically refresh session-related TTLs so they don't expire
        # during long-running turns. Without this, turns exceeding stream_ttl
        # (default 1h) lose their "running" status and stream data, making
        # the session invisible to the resume endpoint (empty on page reload).
        # Both meta key AND stream key are refreshed: the stream key's expire
        # above only fires when publish_chunk is called, but during long
        # sub-agent gaps (task_progress events don't produce chunks), neither
        # key gets refreshed.
        if session_id:
            now = time.perf_counter()
            last_refresh = _meta_ttl_refresh_at.get(session_id, 0)
            if now - last_refresh >= _META_TTL_REFRESH_INTERVAL:
                meta_key = _get_session_meta_key(session_id)
                await redis.expire(meta_key, config.stream_ttl)
                await redis.expire(stream_key, config.stream_ttl)
                _meta_ttl_refresh_at[session_id] = now

        total_time = (time.perf_counter() - start_time) * 1000
        # Only log timing for significant chunks or slow operations
        if (
            chunk_type
            in (
                "StreamStart",
                "StreamFinish",
                "StreamTextStart",
                "StreamTextEnd",
                "StreamToolInputAvailable",
                "StreamToolOutputAvailable",
            )
            or total_time > 50
        ):
            logger.info(
                f"[TIMING] publish_chunk {chunk_type} in {total_time:.1f}ms (xadd={xadd_time:.1f}ms)",
                extra={
                    "json_fields": {
                        **log_meta,
                        "total_time_ms": total_time,
                        "xadd_time_ms": xadd_time,
                        "message_id": message_id,
                    }
                },
            )
    except Exception as e:
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"[TIMING] Failed to publish chunk {chunk_type} after {elapsed:.1f}ms: {e}",
            extra={"json_fields": {**log_meta, "elapsed_ms": elapsed, "error": str(e)}},
            exc_info=True,
        )

    return message_id


async def stream_and_publish(
    session_id: str,
    turn_id: str,
    stream: AsyncIterator[StreamBaseResponse],
) -> AsyncIterator[StreamBaseResponse]:
    """Wrap an async stream iterator with registry publishing.

    Publishes each chunk to the stream registry for frontend SSE consumption,
    skipping ``StreamFinish`` and ``StreamError`` (which are published by
    :func:`mark_session_completed`).

    This is a pass-through: every event from *stream* is yielded unchanged so
    the caller can still consume and aggregate them.  The caller is responsible
    for calling :func:`create_session` before and :func:`mark_session_completed`
    after iterating.

    Args:
        session_id: Chat session ID (for logging only).
        turn_id: Turn UUID that identifies the Redis stream to publish to.
            If empty, publishing is silently skipped (graceful degradation).
        stream: The underlying async iterator of stream events.

    Yields:
        Every event from *stream*, unchanged.
    """
    publish_failed_once = False

    async for event in stream:
        if turn_id and not isinstance(event, (StreamFinish, StreamError)):
            try:
                await publish_chunk(turn_id, event, session_id=session_id)
            except (RedisError, ConnectionError, OSError):
                if not publish_failed_once:
                    publish_failed_once = True
                    logger.warning(
                        "[stream_and_publish] Failed to publish chunk %s for %s "
                        "(further failures logged at DEBUG)",
                        type(event).__name__,
                        session_id[:12],
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "[stream_and_publish] Failed to publish chunk %s",
                        type(event).__name__,
                        exc_info=True,
                    )
        yield event


async def subscribe_to_session(
    session_id: str,
    user_id: str | None,
    last_message_id: str = "0-0",
) -> asyncio.Queue[StreamBaseResponse] | None:
    """Subscribe to a session's stream with replay of missed messages.

    This is fully stateless - uses Redis Stream for replay and pub/sub for live updates.

    Args:
        session_id: Session ID to subscribe to
        user_id: User ID for ownership validation
        last_message_id: Last Redis Stream message ID received ("0-0" for full replay)

    Returns:
        An asyncio Queue that will receive stream chunks, or None if session not found
        or user doesn't have access
    """
    start_time = time.perf_counter()

    # Build log metadata
    log_meta = {"component": "StreamRegistry", "session_id": session_id}
    if user_id:
        log_meta["user_id"] = user_id

    logger.info(
        f"[TIMING] subscribe_to_session STARTED, session={session_id}, user={user_id}, last_msg={last_message_id}",
        extra={"json_fields": {**log_meta, "last_message_id": last_message_id}},
    )

    redis_start = time.perf_counter()
    redis = await get_redis_async()
    meta_key = _get_session_meta_key(session_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]
    hgetall_time = (time.perf_counter() - redis_start) * 1000
    logger.info(
        f"[TIMING] Redis hgetall took {hgetall_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": hgetall_time}},
    )

    # RACE CONDITION FIX: If session not found, retry with backoff.
    # Duplicate requests skip create_session and subscribe immediately; the
    # original request's create_session (a Redis hset) may not have completed
    # yet. 3 × 100ms gives a 300ms window which covers DB-write latency on the
    # original request before the hset even starts.
    if not meta:
        _max_retries = 3
        _retry_delay = 0.1  # 100ms per attempt
        for attempt in range(_max_retries):
            logger.warning(
                f"[TIMING] Session not found (attempt {attempt + 1}/{_max_retries}), "
                f"retrying after {int(_retry_delay * 1000)}ms",
                extra={"json_fields": {**log_meta, "attempt": attempt + 1}},
            )
            await asyncio.sleep(_retry_delay)
            meta = await redis.hgetall(meta_key)  # type: ignore[misc]
            if meta:
                logger.info(
                    f"[TIMING] Session found after {attempt + 1} retries",
                    extra={"json_fields": {**log_meta, "attempts": attempt + 1}},
                )
                break
        else:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[TIMING] Session still not found in Redis after {_max_retries} retries "
                f"({elapsed:.1f}ms total)",
                extra={
                    "json_fields": {
                        **log_meta,
                        "elapsed_ms": elapsed,
                        "reason": "session_not_found_after_retry",
                    }
                },
            )
            return None

    # Note: Redis client uses decode_responses=True, so keys are strings
    session_status = meta.get("status", "")
    session_user_id = meta.get("user_id", "") or None
    log_meta["session_id"] = meta.get("session_id", "")

    # Validate ownership - if session has an owner, requester must match
    if session_user_id:
        if user_id != session_user_id:
            logger.warning(
                f"[TIMING] Access denied: user {user_id} tried to access session owned by {session_user_id}",
                extra={
                    "json_fields": {
                        **log_meta,
                        "session_owner": session_user_id,
                        "reason": "access_denied",
                    }
                },
            )
            return None

    session = _parse_session_meta(meta, session_id)
    subscriber_queue: asyncio.Queue[StreamBaseResponse] = asyncio.Queue()
    stream_key = _get_turn_stream_key(session.turn_id)

    # Step 1: Replay messages from Redis Stream
    xread_start = time.perf_counter()
    messages = await redis.xread({stream_key: last_message_id}, block=None, count=1000)
    xread_time = (time.perf_counter() - xread_start) * 1000
    logger.info(
        f"[TIMING] Redis xread (replay) took {xread_time:.1f}ms, status={session_status}",
        extra={
            "json_fields": {
                **log_meta,
                "duration_ms": xread_time,
                "session_status": session_status,
            }
        },
    )

    replayed_count = 0
    replay_last_id = last_message_id
    if messages:
        for _stream_name, stream_messages in messages:
            for msg_id, msg_data in stream_messages:
                replay_last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                # Note: Redis client uses decode_responses=True, so keys are strings
                if "data" in msg_data:
                    try:
                        chunk_data = orjson.loads(msg_data["data"])
                        chunk = _reconstruct_chunk(chunk_data)
                        if chunk:
                            await subscriber_queue.put(chunk)
                            replayed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to replay message: {e}")

    logger.info(
        f"[TIMING] Replayed {replayed_count} messages, last_id={replay_last_id}",
        extra={
            "json_fields": {
                **log_meta,
                "n_messages_replayed": replayed_count,
                "replay_last_id": replay_last_id,
            }
        },
    )

    # Step 2: If session is still running, start stream listener for live updates
    if session_status == "running":
        logger.info(
            "[TIMING] Session still running, starting _stream_listener",
            extra={"json_fields": {**log_meta, "session_status": session_status}},
        )
        listener_task = asyncio.create_task(
            _stream_listener(
                session_id, subscriber_queue, replay_last_id, log_meta, session.turn_id
            )
        )
        # Track listener task for cleanup on unsubscribe
        _listener_sessions[id(subscriber_queue)] = (session_id, listener_task)
    else:
        # Session is completed/failed - add finish marker
        logger.info(
            f"[TIMING] Session already {session_status}, adding StreamFinish",
            extra={"json_fields": {**log_meta, "session_status": session_status}},
        )
        await subscriber_queue.put(StreamFinish())

    total_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"[TIMING] subscribe_to_session COMPLETED in {total_time:.1f}ms; session={session_id}, "
        f"n_messages_replayed={replayed_count}",
        extra={
            "json_fields": {
                **log_meta,
                "total_time_ms": total_time,
                "n_messages_replayed": replayed_count,
            }
        },
    )
    return subscriber_queue


async def _stream_listener(
    session_id: str,
    subscriber_queue: asyncio.Queue[StreamBaseResponse],
    last_replayed_id: str,
    log_meta: dict | None = None,
    turn_id: str = "",
) -> None:
    """Listen to Redis Stream for new messages using blocking XREAD.

    This approach avoids the duplicate message issue that can occur with pub/sub
    when messages are published during the gap between replay and subscription.

    Args:
        session_id: Session ID to listen for
        subscriber_queue: Queue to deliver messages to
        last_replayed_id: Last message ID from replay (continue from here)
        log_meta: Structured logging metadata
        turn_id: Per-turn UUID for stream key resolution
    """
    start_time = time.perf_counter()

    # Use provided log_meta or build minimal one
    if log_meta is None:
        log_meta = {"component": "StreamRegistry", "session_id": session_id}

    logger.info(
        f"[TIMING] _stream_listener STARTED, session={session_id}, last_id={last_replayed_id}",
        extra={"json_fields": {**log_meta, "last_replayed_id": last_replayed_id}},
    )

    queue_id = id(subscriber_queue)
    # Track the last successfully delivered message ID for recovery hints
    last_delivered_id = last_replayed_id
    messages_delivered = 0
    first_message_time = None
    xread_count = 0

    try:
        redis = await get_redis_async()
        stream_key = _get_turn_stream_key(turn_id)
        current_id = last_replayed_id

        while True:
            # Block for up to 5 seconds waiting for new messages
            # This allows periodic checking if session is still running
            # Short timeout prevents frontend timeout (12s) while waiting for heartbeats (15s)
            xread_start = time.perf_counter()
            xread_count += 1
            messages = await redis.xread(
                {stream_key: current_id}, block=5000, count=100
            )
            xread_time = (time.perf_counter() - xread_start) * 1000

            if messages:
                msg_count = sum(len(msgs) for _, msgs in messages)
                logger.info(
                    f"[TIMING] xread #{xread_count} returned {msg_count} messages in {xread_time:.1f}ms",
                    extra={
                        "json_fields": {
                            **log_meta,
                            "xread_count": xread_count,
                            "n_messages": msg_count,
                            "duration_ms": xread_time,
                        }
                    },
                )
            elif xread_time > 1000:
                # Only log timeouts (30s blocking)
                logger.info(
                    f"[TIMING] xread #{xread_count} timeout after {xread_time:.1f}ms",
                    extra={
                        "json_fields": {
                            **log_meta,
                            "xread_count": xread_count,
                            "duration_ms": xread_time,
                            "reason": "timeout",
                        }
                    },
                )

            if not messages:
                # Timeout - check if session is still running
                meta_key = _get_session_meta_key(session_id)
                status = await redis.hget(meta_key, "status")  # type: ignore[misc]
                # Stop if session metadata is gone (TTL expired) or status is not "running"
                if status != "running":
                    try:
                        await asyncio.wait_for(
                            subscriber_queue.put(StreamFinish()),
                            timeout=QUEUE_PUT_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout delivering finish event for session {session_id}"
                        )
                    break
                # Session still running - send heartbeat to keep connection alive
                # This prevents frontend timeout (12s) during long-running operations
                try:
                    await asyncio.wait_for(
                        subscriber_queue.put(StreamHeartbeat()),
                        timeout=QUEUE_PUT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout delivering heartbeat for session {session_id}"
                    )
                continue

            for _stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    current_id = msg_id if isinstance(msg_id, str) else msg_id.decode()

                    if "data" not in msg_data:
                        continue

                    try:
                        chunk_data = orjson.loads(msg_data["data"])
                        chunk = _reconstruct_chunk(chunk_data)
                        if chunk:
                            try:
                                await asyncio.wait_for(
                                    subscriber_queue.put(chunk),
                                    timeout=QUEUE_PUT_TIMEOUT,
                                )
                                # Update last delivered ID on successful delivery
                                last_delivered_id = current_id
                                messages_delivered += 1
                                if first_message_time is None:
                                    first_message_time = time.perf_counter()
                                    elapsed = (first_message_time - start_time) * 1000
                                    logger.info(
                                        f"[TIMING] FIRST live message at {elapsed:.1f}ms, type={type(chunk).__name__}",
                                        extra={
                                            "json_fields": {
                                                **log_meta,
                                                "elapsed_ms": elapsed,
                                                "chunk_type": type(chunk).__name__,
                                            }
                                        },
                                    )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"[TIMING] Subscriber queue full, delivery timed out after {QUEUE_PUT_TIMEOUT}s",
                                    extra={
                                        "json_fields": {
                                            **log_meta,
                                            "timeout_s": QUEUE_PUT_TIMEOUT,
                                            "reason": "queue_full",
                                        }
                                    },
                                )
                                # Send overflow error with recovery info
                                try:
                                    overflow_error = StreamError(
                                        errorText="Message delivery timeout - some messages may have been missed",
                                        code="QUEUE_OVERFLOW",
                                        details={
                                            "last_delivered_id": last_delivered_id,
                                            "recovery_hint": f"Reconnect with last_message_id={last_delivered_id}",
                                        },
                                    )
                                    subscriber_queue.put_nowait(overflow_error)
                                except asyncio.QueueFull:
                                    # Queue is completely stuck, nothing more we can do
                                    logger.error(
                                        f"Cannot deliver overflow error for session {session_id}, "
                                        "queue completely blocked"
                                    )

                            # Stop listening on finish
                            if isinstance(chunk, StreamFinish):
                                total_time = (time.perf_counter() - start_time) * 1000
                                logger.info(
                                    f"[TIMING] StreamFinish received in {total_time / 1000:.1f}s; delivered={messages_delivered}",
                                    extra={
                                        "json_fields": {
                                            **log_meta,
                                            "total_time_ms": total_time,
                                            "messages_delivered": messages_delivered,
                                        }
                                    },
                                )
                                return
                    except Exception as e:
                        logger.warning(
                            f"Error processing stream message: {e}",
                            extra={"json_fields": {**log_meta, "error": str(e)}},
                        )

    except asyncio.CancelledError:
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[TIMING] _stream_listener CANCELLED after {elapsed:.1f}ms, delivered={messages_delivered}",
            extra={
                "json_fields": {
                    **log_meta,
                    "elapsed_ms": elapsed,
                    "messages_delivered": messages_delivered,
                    "reason": "cancelled",
                }
            },
        )
        raise  # Re-raise to propagate cancellation
    except Exception as e:
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"[TIMING] _stream_listener ERROR after {elapsed:.1f}ms: {e}",
            extra={"json_fields": {**log_meta, "elapsed_ms": elapsed, "error": str(e)}},
        )
        # On error, send finish to unblock subscriber
        try:
            await asyncio.wait_for(
                subscriber_queue.put(StreamFinish()),
                timeout=QUEUE_PUT_TIMEOUT,
            )
        except (asyncio.TimeoutError, asyncio.QueueFull):
            logger.warning(
                "Could not deliver finish event after error",
                extra={"json_fields": log_meta},
            )
    finally:
        # Clean up listener session mapping on exit
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[TIMING] _stream_listener FINISHED in {total_time / 1000:.1f}s; session={session_id}, "
            f"delivered={messages_delivered}, xread_count={xread_count}",
            extra={
                "json_fields": {
                    **log_meta,
                    "total_time_ms": total_time,
                    "messages_delivered": messages_delivered,
                    "xread_count": xread_count,
                }
            },
        )
        _listener_sessions.pop(queue_id, None)


async def mark_session_completed(
    session_id: str,
    error_message: str | None = None,
    *,
    skip_error_publish: bool = False,
) -> bool:
    """Mark a session as completed, then publish StreamFinish.

    This is the SINGLE place that publishes StreamFinish to the turn stream.
    Services must NOT yield StreamFinish themselves — the processor intercepts
    it and calls this function instead, ensuring status is set before
    StreamFinish reaches the frontend.

    Uses atomic compare-and-swap via Lua script to prevent race conditions.
    Idempotent — calling multiple times is safe (returns False on no-op).

    Args:
        session_id: Session ID to mark as completed
        error_message: If provided, marks as "failed" and publishes a
            StreamError before StreamFinish. Otherwise marks as "completed".
        skip_error_publish: If True, still marks the session as "failed" but
            does NOT publish a StreamError event. Use this when the error has
            already been published to the stream (e.g. via stream_and_publish)
            to avoid duplicate error delivery to the frontend.

    Returns:
        True if session was newly marked completed, False if already completed/failed
    """
    status: Literal["completed", "failed"] = "failed" if error_message else "completed"
    redis = await get_redis_async()
    meta_key = _get_session_meta_key(session_id)

    # Resolve turn_id for publishing to the correct stream
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]
    turn_id = _parse_session_meta(meta, session_id).turn_id if meta else session_id

    # Atomic compare-and-swap: only update if status is "running"
    swapped = await hash_compare_and_set(
        redis, meta_key, "status", expected="running", new=status
    )

    # Clean up the in-memory TTL refresh tracker to prevent unbounded growth.
    _meta_ttl_refresh_at.pop(session_id, None)

    if not swapped:
        logger.debug(f"Session {session_id} already completed/failed, skipping")
        return False

    if error_message and not skip_error_publish:
        try:
            await publish_chunk(turn_id, StreamError(errorText=error_message))
        except Exception as e:
            logger.warning(
                f"Failed to publish error event for session {session_id}: {e}"
            )

    # Compute wall-clock duration from session created_at.
    # Only persist when (a) the session completed successfully and
    # (b) created_at was actually present in Redis meta (not a fallback).
    duration_ms: int | None = None
    if meta and not error_message:
        created_at_raw = meta.get("created_at")
        if created_at_raw:
            try:
                created_at = datetime.fromisoformat(str(created_at_raw))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                elapsed = datetime.now(timezone.utc) - created_at
                duration_ms = max(0, int(elapsed.total_seconds() * 1000))
            except (ValueError, TypeError):
                logger.warning(
                    "Failed to compute session duration for %s (created_at=%r)",
                    session_id,
                    created_at_raw,
                )

    # Persist duration on the last assistant message
    if duration_ms is not None:
        try:
            await chat_db().set_turn_duration(session_id, duration_ms)
        except Exception as e:
            logger.warning(f"Failed to save turn duration for {session_id}: {e}")

    # Publish StreamFinish AFTER status is set to "completed"/"failed".
    # This is the SINGLE place that publishes StreamFinish — services and
    # the processor must NOT publish it themselves.
    try:
        await publish_chunk(
            turn_id,
            StreamFinish(),
        )
    except Exception as e:
        logger.error(
            f"Failed to publish StreamFinish for session {session_id}: {e}. "
            "The _stream_listener will detect completion via status polling."
        )

    # Clean up local session reference if exists
    _local_sessions.pop(session_id, None)

    # Publish copilot completion notification via WebSocket
    if meta:
        parsed = _parse_session_meta(meta, session_id)
        if parsed.user_id:
            try:
                await _notification_bus.publish(
                    NotificationEvent(
                        user_id=parsed.user_id,
                        payload=CopilotCompletionPayload(
                            type="copilot_completion",
                            event="session_completed",
                            session_id=session_id,
                            status=status,
                        ),
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to publish copilot completion notification "
                    f"for session {session_id}: {e}"
                )

    return True


async def get_session(session_id: str) -> ActiveSession | None:
    """Get a session by its ID from Redis.

    Args:
        session_id: Session ID to look up

    Returns:
        ActiveSession if found, None otherwise
    """
    redis = await get_redis_async()
    meta_key = _get_session_meta_key(session_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        return None

    return _parse_session_meta(meta, session_id)


async def get_session_with_expiry_info(
    session_id: str,
) -> tuple[ActiveSession | None, str | None]:
    """Get a session by its ID with expiration detection.

    Returns (session, error_code) where error_code is:
    - None if session found
    - "SESSION_EXPIRED" if stream exists but metadata is gone (TTL expired)
    - "SESSION_NOT_FOUND" if neither exists

    Args:
        session_id: Session ID to look up

    Returns:
        Tuple of (ActiveSession or None, error_code or None)
    """
    redis = await get_redis_async()
    meta_key = _get_session_meta_key(session_id)

    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        # Metadata expired — we can't resolve turn_id, so check using
        # session_id as a best-effort fallback for the stream key.
        stream_key = _get_turn_stream_key(session_id)
        stream_len = await redis.xlen(stream_key)
        if stream_len > 0:
            return None, "SESSION_EXPIRED"
        return None, "SESSION_NOT_FOUND"

    return _parse_session_meta(meta, session_id), None


async def get_active_session(
    session_id: str,
    user_id: str | None = None,
) -> tuple[ActiveSession | None, str]:
    """Get the active (running) session, if any.

    Direct O(1) lookup by session_id.

    Args:
        session_id: Session ID to look up
        user_id: User ID for ownership validation (optional)

    Returns:
        Tuple of (ActiveSession if found and running, last_message_id from Redis Stream)
    """

    redis = await get_redis_async()
    meta_key = _get_session_meta_key(session_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        return None, "0-0"

    session_status = meta.get("status", "")
    session_user_id = meta.get("user_id", "") or None

    if session_status != "running":
        return None, "0-0"

    # Validate ownership - if session has an owner, requester must match
    if session_user_id and user_id != session_user_id:
        return None, "0-0"

    # Check if session is stale (running beyond tool timeout + buffer).
    # Auto-complete it to prevent infinite polling loops.
    # Synchronous tools can run up to COPILOT_CONSUMER_TIMEOUT_SECONDS (1 hour),
    # so we add a 5-minute buffer to avoid false positives during legitimate operations.
    created_at_str = meta.get("created_at")
    if created_at_str:
        try:
            created_at = datetime.fromisoformat(created_at_str)
            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            stale_threshold = COPILOT_CONSUMER_TIMEOUT_SECONDS + 300  # + 5min buffer
            if age_seconds > stale_threshold:
                logger.warning(
                    f"[STALE_SESSION] Auto-completing stale session {session_id[:8]}... "
                    f"(running for {age_seconds:.0f}s, threshold: {stale_threshold}s)"
                )
                await mark_session_completed(
                    session_id,
                    error_message=f"Session timed out after {age_seconds:.0f}s",
                )
                return None, "0-0"
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse created_at: {e}")

    session = _parse_session_meta(meta, session_id)
    logger.info(
        f"[SESSION_LOOKUP] Found running session {session_id[:8]}..., turn_id={session.turn_id[:8]}"
    )

    # Get the last message ID from Redis Stream (keyed by turn_id)
    stream_key = _get_turn_stream_key(session.turn_id)
    last_id = "0-0"
    try:
        messages = await redis.xrevrange(stream_key, count=1)
        if messages:
            msg_id = messages[0][0]
            last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
    except Exception as e:
        logger.warning(f"Failed to get last message ID: {e}")

    return session, last_id


def _reconstruct_chunk(chunk_data: dict) -> StreamBaseResponse | None:
    """Reconstruct a StreamBaseResponse from JSON data.

    Args:
        chunk_data: Parsed JSON data from Redis

    Returns:
        Reconstructed response object, or None if unknown type
    """
    # Map response types to their corresponding classes
    type_to_class: dict[str, type[StreamBaseResponse]] = {
        ResponseType.START.value: StreamStart,
        ResponseType.FINISH.value: StreamFinish,
        ResponseType.START_STEP.value: StreamStartStep,
        ResponseType.FINISH_STEP.value: StreamFinishStep,
        ResponseType.TEXT_START.value: StreamTextStart,
        ResponseType.TEXT_DELTA.value: StreamTextDelta,
        ResponseType.TEXT_END.value: StreamTextEnd,
        ResponseType.REASONING_START.value: StreamReasoningStart,
        ResponseType.REASONING_DELTA.value: StreamReasoningDelta,
        ResponseType.REASONING_END.value: StreamReasoningEnd,
        ResponseType.TOOL_INPUT_START.value: StreamToolInputStart,
        ResponseType.TOOL_INPUT_AVAILABLE.value: StreamToolInputAvailable,
        ResponseType.TOOL_OUTPUT_AVAILABLE.value: StreamToolOutputAvailable,
        ResponseType.ERROR.value: StreamError,
        ResponseType.USAGE.value: StreamUsage,
        ResponseType.HEARTBEAT.value: StreamHeartbeat,
    }

    chunk_type = chunk_data.get("type")
    chunk_class = type_to_class.get(chunk_type)  # type: ignore[arg-type]

    if chunk_class is None:
        logger.warning(f"Unknown chunk type: {chunk_type}")
        return None

    try:
        return chunk_class(**chunk_data)
    except Exception as e:
        logger.warning(f"Failed to reconstruct chunk of type {chunk_type}: {e}")
        return None


async def set_session_asyncio_task(session_id: str, asyncio_task: asyncio.Task) -> None:
    """Track the asyncio.Task for a session (local reference only).

    This is just for cleanup purposes - the session state is in Redis.

    Args:
        session_id: Session ID
        asyncio_task: The asyncio Task to track
    """
    _local_sessions[session_id] = asyncio_task


async def unsubscribe_from_session(
    session_id: str,
    subscriber_queue: asyncio.Queue[StreamBaseResponse],
) -> None:
    """Clean up when a subscriber disconnects.

    Cancels the XREAD-based listener task associated with this subscriber queue
    to prevent resource leaks.

    Args:
        session_id: Session ID
        subscriber_queue: The subscriber's queue used to look up the listener task
    """
    queue_id = id(subscriber_queue)
    listener_entry = _listener_sessions.pop(queue_id, None)

    if listener_entry is None:
        logger.debug(
            f"No listener task found for session {session_id} queue {queue_id} "
            "(may have already completed)"
        )
        return

    stored_session_id, listener_task = listener_entry

    if stored_session_id != session_id:
        logger.warning(
            f"Session ID mismatch in unsubscribe: expected {session_id}, "
            f"found {stored_session_id}"
        )

    if listener_task.done():
        logger.debug(f"Listener task for session {session_id} already completed")
        return

    # Cancel the listener task
    listener_task.cancel()

    try:
        # Wait for the task to be cancelled with a timeout
        await asyncio.wait_for(listener_task, timeout=5.0)
    except asyncio.CancelledError:
        # Expected - the task was successfully cancelled
        pass
    except asyncio.TimeoutError:
        logger.warning(
            f"Timeout waiting for listener task cancellation for session {session_id}"
        )
    except Exception as e:
        logger.error(
            f"Error during listener task cancellation for session {session_id}: {e}"
        )

    logger.debug(f"Successfully unsubscribed from session {session_id}")


async def disconnect_all_listeners(session_id: str) -> int:
    """Cancel every active listener task for *session_id*.

    Called when the frontend switches away from a session and wants the
    backend to release resources immediately rather than waiting for the
    XREAD timeout.

    Scope / limitations (best-effort optimisation, not a correctness primitive):
    - Pod-local: ``_listener_sessions`` is in-memory. If the DELETE request
      lands on a different worker than the one serving the SSE, no listener
      is cancelled here — the SSE worker still releases on its XREAD timeout.
    - Session-scoped (not subscriber-scoped): cancels every active listener
      for the session on this pod. In the rare case a single user opens two
      SSE connections to the same session on the same pod (e.g. two tabs),
      both would be torn down. Cross-pod, subscriber-scoped cancellation
      would require a Redis pub/sub fan-out with per-listener tokens; that
      is not implemented here because the XREAD timeout already bounds the
      worst case.

    Returns the number of listener tasks that were cancelled.
    """
    to_cancel: list[tuple[int, asyncio.Task]] = [
        (qid, task)
        for qid, (sid, task) in list(_listener_sessions.items())
        if sid == session_id and not task.done()
    ]

    for qid, task in to_cancel:
        _listener_sessions.pop(qid, None)
        task.cancel()

    cancelled = 0
    for _qid, task in to_cancel:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            cancelled += 1
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Error cancelling listener for session {session_id}: {e}")

    if cancelled:
        logger.info(f"Disconnected {cancelled} listener(s) for session {session_id}")
    return cancelled
