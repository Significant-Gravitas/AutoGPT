"""Stream registry for managing reconnectable SSE streams.

This module provides a registry for tracking active streaming tasks and their
messages. It uses Redis for all state management (no in-memory state), making
pods stateless and horizontally scalable.

Architecture:
- Redis Stream: Persists all messages for replay and real-time delivery
- Redis Hash: Task metadata (status, session_id, etc.)

Subscribers:
1. Replay missed messages from Redis Stream (XREAD)
2. Listen for live updates via blocking XREAD
3. No in-memory state required on the subscribing pod
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import orjson

from backend.data.redis_client import get_redis_async

from .config import ChatConfig
from .response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamHeartbeat,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

# Track background tasks for this pod (just the asyncio.Task reference, not subscribers)
_local_tasks: dict[str, asyncio.Task] = {}

# Track listener tasks per subscriber queue for cleanup
# Maps queue id() to (session_id, asyncio.Task) for proper cleanup on unsubscribe
_listener_tasks: dict[int, tuple[str, asyncio.Task]] = {}

# Timeout for putting chunks into subscriber queues (seconds)
# If the queue is full and doesn't drain within this time, send an overflow error
QUEUE_PUT_TIMEOUT = 5.0

# Lua script for atomic compare-and-swap status update (idempotent completion)
# Returns 1 if status was updated, 0 if already completed/failed
COMPLETE_TASK_SCRIPT = """
local current = redis.call("HGET", KEYS[1], "status")
if current == "running" then
    redis.call("HSET", KEYS[1], "status", ARGV[1])
    return 1
end
return 0
"""


@dataclass
class ActiveTask:
    """Represents an active streaming task (metadata only, no in-memory queues)."""

    session_id: str
    user_id: str | None
    tool_call_id: str
    tool_name: str
    operation_id: str
    blocking: bool = False  # If True, HTTP request is waiting for completion
    status: Literal["running", "completed", "failed"] = "running"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    asyncio_task: asyncio.Task | None = None


def _get_task_meta_key(session_id: str) -> str:
    """Get Redis key for task metadata (now keyed by session_id)."""
    return f"{config.task_meta_prefix}{session_id}"


def _get_task_stream_key(session_id: str) -> str:
    """Get Redis key for task message stream (now keyed by session_id)."""
    return f"{config.task_stream_prefix}{session_id}"


def _get_operation_mapping_key(operation_id: str) -> str:
    """Get Redis key for operation_id to task_id mapping."""
    return f"{config.task_op_prefix}{operation_id}"


async def create_task(
    session_id: str,
    user_id: str | None,
    tool_call_id: str,
    tool_name: str,
    operation_id: str,
    blocking: bool = False,
) -> ActiveTask:
    """Create a new streaming task in Redis (keyed by session_id).

    Args:
        session_id: Chat session ID (used as task identifier)
        user_id: User ID (may be None for anonymous)
        tool_call_id: Tool call ID from the LLM
        tool_name: Name of the tool being executed
        operation_id: Operation ID for webhook callbacks

    Returns:
        The created ActiveTask instance (metadata only)
    """
    import time

    start_time = time.perf_counter()

    # Build log metadata for structured logging
    log_meta = {
        "component": "StreamRegistry",
        "session_id": session_id,
    }
    if user_id:
        log_meta["user_id"] = user_id

    logger.info(
        f"[TIMING] create_task STARTED, session={session_id}, user={user_id}",
        extra={"json_fields": log_meta},
    )

    # Create task (use session_id directly, no separate task_id)
    task = ActiveTask(
        session_id=session_id,
        user_id=user_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        operation_id=operation_id,
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

    meta_key = _get_task_meta_key(session_id)
    op_key = _get_operation_mapping_key(operation_id)
    logger.info(f"[DEBUG] create_task storing at key: {meta_key}, status=running")

    hset_start = time.perf_counter()
    await redis.hset(  # type: ignore[misc]
        meta_key,
        mapping={
            "session_id": session_id,
            "user_id": user_id or "",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "operation_id": operation_id,
            "blocking": "1" if blocking else "0",
            "status": task.status,
            "created_at": task.created_at.isoformat(),
        },
    )
    hset_time = (time.perf_counter() - hset_start) * 1000
    logger.info(
        f"[TIMING] redis.hset took {hset_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": hset_time}},
    )

    await redis.expire(meta_key, config.stream_ttl)

    # Create operation_id -> session_id mapping for webhook lookups
    await redis.set(op_key, session_id, ex=config.stream_ttl)

    total_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"[TIMING] create_task COMPLETED in {total_time:.1f}ms; session={session_id}",
        extra={"json_fields": {**log_meta, "total_time_ms": total_time}},
    )

    return task


async def publish_chunk(
    session_id: str,
    chunk: StreamBaseResponse,
) -> str:
    """Publish a chunk to Redis Stream.

    All delivery is via Redis Streams - no in-memory state.

    Args:
        session_id: Session ID to publish to
        chunk: The stream response chunk to publish

    Returns:
        The Redis Stream message ID
    """
    import time

    start_time = time.perf_counter()
    chunk_type = type(chunk).__name__
    chunk_json = chunk.model_dump_json()
    message_id = "0-0"

    # Build log metadata
    log_meta = {
        "component": "StreamRegistry",
        "session_id": session_id,
        "chunk_type": chunk_type,
    }

    try:
        redis = await get_redis_async()
        stream_key = _get_task_stream_key(session_id)

        # DEBUG: Log what we're about to publish
        from .response_model import StreamFinish, StreamTextDelta, StreamTextStart

        if isinstance(chunk, StreamTextDelta):
            logger.info(
                f"[DEBUG_CONVERSATION] [PUBLISH] StreamTextDelta to {session_id[:12]}: {chunk.delta!r}"
            )
        elif isinstance(chunk, StreamTextStart):
            logger.info(
                f"[DEBUG_CONVERSATION] [PUBLISH] StreamTextStart to {session_id[:12]}"
            )
        elif isinstance(chunk, StreamFinish):
            logger.info(
                f"[DEBUG_CONVERSATION] [PUBLISH] StreamFinish to {session_id[:12]}"
            )

        # Write to Redis Stream for persistence and real-time delivery
        xadd_start = time.perf_counter()
        raw_id = await redis.xadd(
            stream_key,
            {"data": chunk_json},
            maxlen=config.stream_max_length,
        )
        xadd_time = (time.perf_counter() - xadd_start) * 1000
        message_id = raw_id if isinstance(raw_id, str) else raw_id.decode()

        # Set TTL on stream to match task metadata TTL
        await redis.expire(stream_key, config.stream_ttl)

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


async def subscribe_to_task(
    session_id: str,
    user_id: str | None,
    last_message_id: str = "0-0",
) -> asyncio.Queue[StreamBaseResponse] | None:
    """Subscribe to a task's stream with replay of missed messages.

    This is fully stateless - uses Redis Stream for replay and pub/sub for live updates.

    Args:
        session_id: Session ID to subscribe to
        user_id: User ID for ownership validation
        last_message_id: Last Redis Stream message ID received ("0-0" for full replay)

    Returns:
        An asyncio Queue that will receive stream chunks, or None if task not found
        or user doesn't have access
    """
    import asyncio
    import time

    start_time = time.perf_counter()

    # Build log metadata
    log_meta = {"component": "StreamRegistry", "session_id": session_id}
    if user_id:
        log_meta["user_id"] = user_id

    logger.info(
        f"[TIMING] subscribe_to_task STARTED, task={session_id}, user={user_id}, last_msg={last_message_id}",
        extra={"json_fields": {**log_meta, "last_message_id": last_message_id}},
    )

    redis_start = time.perf_counter()
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(session_id)
    logger.info(f"[DEBUG] subscribe_to_task looking for key: {meta_key}")
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]
    hgetall_time = (time.perf_counter() - redis_start) * 1000
    logger.info(
        f"[TIMING] Redis hgetall took {hgetall_time:.1f}ms, result={bool(meta)}, keys={list(meta.keys()) if meta else []}",
        extra={"json_fields": {**log_meta, "duration_ms": hgetall_time}},
    )

    # RACE CONDITION FIX: If task not found, retry once after small delay
    # This handles the case where subscribe_to_task is called immediately
    # after create_task but before Redis propagates the write
    if not meta:
        logger.warning(
            "[TIMING] Task not found on first attempt, retrying after 50ms delay",
            extra={"json_fields": {**log_meta}},
        )
        await asyncio.sleep(0.05)  # 50ms
        meta = await redis.hgetall(meta_key)  # type: ignore[misc]
        if not meta:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[TIMING] Task still not found in Redis after retry ({elapsed:.1f}ms total)",
                extra={
                    "json_fields": {
                        **log_meta,
                        "elapsed_ms": elapsed,
                        "reason": "task_not_found_after_retry",
                    }
                },
            )
            return None
        logger.info(
            "[TIMING] Task found after retry",
            extra={"json_fields": {**log_meta}},
        )

    # Note: Redis client uses decode_responses=True, so keys are strings
    task_status = meta.get("status", "")
    task_user_id = meta.get("user_id", "") or None
    log_meta["session_id"] = meta.get("session_id", "")

    # Validate ownership - if task has an owner, requester must match
    if task_user_id:
        if user_id != task_user_id:
            logger.warning(
                f"[TIMING] Access denied: user {user_id} tried to access task owned by {task_user_id}",
                extra={
                    "json_fields": {
                        **log_meta,
                        "task_owner": task_user_id,
                        "reason": "access_denied",
                    }
                },
            )
            return None

    subscriber_queue: asyncio.Queue[StreamBaseResponse] = asyncio.Queue()
    stream_key = _get_task_stream_key(session_id)

    # Step 1: Replay messages from Redis Stream
    xread_start = time.perf_counter()
    messages = await redis.xread({stream_key: last_message_id}, block=None, count=1000)
    xread_time = (time.perf_counter() - xread_start) * 1000
    logger.info(
        f"[TIMING] Redis xread (replay) took {xread_time:.1f}ms, status={task_status}",
        extra={
            "json_fields": {
                **log_meta,
                "duration_ms": xread_time,
                "task_status": task_status,
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
        f"[DEBUG_REPLAY] Replayed {replayed_count} messages from last_message_id={last_message_id} to {replay_last_id}",
        extra={
            "json_fields": {
                **log_meta,
                "n_messages_replayed": replayed_count,
                "replay_last_id": replay_last_id,
                "started_from_id": last_message_id,
            }
        },
    )

    # Step 2: If task is still running, start stream listener for live updates
    if task_status == "running":
        logger.info(
            "[TIMING] Task still running, starting _stream_listener",
            extra={"json_fields": {**log_meta, "task_status": task_status}},
        )
        listener_task = asyncio.create_task(
            _stream_listener(session_id, subscriber_queue, replay_last_id, log_meta)
        )
        # Track listener task for cleanup on unsubscribe
        _listener_tasks[id(subscriber_queue)] = (session_id, listener_task)
    else:
        # Task is completed/failed - add finish marker
        logger.info(
            f"[TIMING] Task already {task_status}, adding StreamFinish",
            extra={"json_fields": {**log_meta, "task_status": task_status}},
        )
        await subscriber_queue.put(StreamFinish())

    total_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"[TIMING] subscribe_to_task COMPLETED in {total_time:.1f}ms; task={session_id}, "
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
) -> None:
    """Listen to Redis Stream for new messages using blocking XREAD.

    This approach avoids the duplicate message issue that can occur with pub/sub
    when messages are published during the gap between replay and subscription.

    Args:
        session_id: Session ID to listen for
        subscriber_queue: Queue to deliver messages to
        last_replayed_id: Last message ID from replay (continue from here)
        log_meta: Structured logging metadata
    """
    import time

    start_time = time.perf_counter()

    # Use provided log_meta or build minimal one
    if log_meta is None:
        log_meta = {"component": "StreamRegistry", "session_id": session_id}

    logger.info(
        f"[TIMING] _stream_listener STARTED, task={session_id}, last_id={last_replayed_id}",
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
        stream_key = _get_task_stream_key(session_id)
        current_id = last_replayed_id

        while True:
            # Block for up to 5 seconds waiting for new messages
            # This allows periodic checking if task is still running
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
                # Timeout - check if task is still running
                meta_key = _get_task_meta_key(session_id)
                status = await redis.hget(meta_key, "status")  # type: ignore[misc]
                if status and status != "running":
                    try:
                        await asyncio.wait_for(
                            subscriber_queue.put(StreamFinish()),
                            timeout=QUEUE_PUT_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout delivering finish event for task {session_id}"
                        )
                    break
                # Task still running - send heartbeat to keep connection alive
                # This prevents frontend timeout (12s) during long-running operations
                try:
                    await asyncio.wait_for(
                        subscriber_queue.put(StreamHeartbeat()),
                        timeout=QUEUE_PUT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout delivering heartbeat for task {session_id}"
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
                                        f"Cannot deliver overflow error for task {session_id}, "
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
        # Clean up listener task mapping on exit
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[TIMING] _stream_listener FINISHED in {total_time / 1000:.1f}s; task={session_id}, "
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
        _listener_tasks.pop(queue_id, None)


async def mark_task_completed(
    session_id: str,
    status: Literal["completed", "failed"] = "completed",
    *,
    error_message: str | None = None,
) -> bool:
    """Mark a task as completed and publish finish event.

    This is idempotent - calling multiple times with the same task_id is safe.
    Uses atomic compare-and-swap via Lua script to prevent race conditions.
    Status is updated first (source of truth), then finish event is published (best-effort).

    Args:
        session_id: Session ID to mark as completed
        status: Final status ("completed" or "failed")
        error_message: If provided and status="failed", publish a StreamError
            before StreamFinish so connected clients see why the task ended.
            If not provided, no StreamError is published (caller should publish
            manually if needed to avoid duplicates).

    Returns:
        True if task was newly marked completed, False if already completed/failed
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(session_id)

    # Atomic compare-and-swap: only update if status is "running"
    # This prevents race conditions when multiple callers try to complete simultaneously
    result = await redis.eval(COMPLETE_TASK_SCRIPT, 1, meta_key, status)  # type: ignore[misc]

    if result == 0:
        logger.debug(f"Task {session_id} already completed/failed, skipping")
        return False

    # Publish error event before finish so connected clients know WHY the
    # task ended. Only publish if caller provided an explicit error message
    # to avoid duplicates with code paths that manually publish StreamError.
    # This is best-effort — if it fails, the StreamFinish still ensures
    # listeners clean up.
    if status == "failed" and error_message:
        try:
            await publish_chunk(session_id, StreamError(errorText=error_message))
        except Exception as e:
            logger.warning(f"Failed to publish error event for task {session_id}: {e}")

    # THEN publish finish event (best-effort - listeners can detect via status polling)
    try:
        await publish_chunk(session_id, StreamFinish())
    except Exception as e:
        logger.error(
            f"Failed to publish finish event for task {session_id}: {e}. "
            "Listeners will detect completion via status polling."
        )

    # Clean up local task reference if exists
    _local_tasks.pop(session_id, None)
    return True


async def find_task_by_operation_id(operation_id: str) -> ActiveTask | None:
    """Find a task by its operation ID.

    Used by webhook callbacks to locate the task to update.

    Args:
        operation_id: Operation ID to search for

    Returns:
        ActiveTask if found, None otherwise
    """
    redis = await get_redis_async()
    op_key = _get_operation_mapping_key(operation_id)
    session_id = await redis.get(op_key)

    if not session_id:
        return None

    session_id_str = (
        session_id.decode() if isinstance(session_id, bytes) else session_id
    )
    return await get_task(session_id_str)


async def get_task(session_id: str) -> ActiveTask | None:
    """Get a task by its ID from Redis.

    Args:
        session_id: Session ID to look up

    Returns:
        ActiveTask if found, None otherwise
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(session_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        return None

    # Note: Redis client uses decode_responses=True, so keys/values are strings
    return ActiveTask(
        session_id=meta.get("session_id", ""),
        user_id=meta.get("user_id", "") or None,
        tool_call_id=meta.get("tool_call_id", ""),
        tool_name=meta.get("tool_name", ""),
        operation_id=meta.get("operation_id", ""),
        blocking=meta.get("blocking") == "1",
        status=meta.get("status", "running"),  # type: ignore[arg-type]
    )


async def get_task_with_expiry_info(
    session_id: str,
) -> tuple[ActiveTask | None, str | None]:
    """Get a task by its ID with expiration detection.

    Returns (task, error_code) where error_code is:
    - None if task found
    - "TASK_EXPIRED" if stream exists but metadata is gone (TTL expired)
    - "TASK_NOT_FOUND" if neither exists

    Args:
        session_id: Session ID to look up

    Returns:
        Tuple of (ActiveTask or None, error_code or None)
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(session_id)
    stream_key = _get_task_stream_key(session_id)

    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        # Check if stream still has data (metadata expired but stream hasn't)
        stream_len = await redis.xlen(stream_key)
        if stream_len > 0:
            return None, "TASK_EXPIRED"
        return None, "TASK_NOT_FOUND"

    # Note: Redis client uses decode_responses=True, so keys/values are strings
    return (
        ActiveTask(
            session_id=meta.get("session_id", ""),
            user_id=meta.get("user_id", "") or None,
            tool_call_id=meta.get("tool_call_id", ""),
            tool_name=meta.get("tool_name", ""),
            operation_id=meta.get("operation_id", ""),
            blocking=meta.get("blocking") == "1",
            status=meta.get("status", "running"),  # type: ignore[arg-type]
        ),
        None,
    )


async def get_active_task_for_session(
    session_id: str,
    user_id: str | None = None,
) -> tuple[ActiveTask | None, str]:
    """Get the active (running) task for a session, if any.

    Scans Redis for tasks matching the session_id with status="running".

    Args:
        session_id: Session ID to look up
        user_id: User ID for ownership validation (optional)

    Returns:
        Tuple of (ActiveTask if found and running, last_message_id from Redis Stream)
    """

    redis = await get_redis_async()

    # Scan Redis for task metadata keys
    cursor = 0
    tasks_checked = 0

    while True:
        cursor, keys = await redis.scan(
            cursor, match=f"{config.task_meta_prefix}*", count=100
        )

        for key in keys:
            tasks_checked += 1
            meta: dict[Any, Any] = await redis.hgetall(key)  # type: ignore[misc]
            if not meta:
                continue

            # Note: Redis client uses decode_responses=True, so keys/values are strings
            task_session_id = meta.get("session_id", "")
            task_status = meta.get("status", "")
            task_user_id = meta.get("user_id", "") or None

            if task_session_id == session_id and task_status == "running":
                # Validate ownership - if task has an owner, requester must match
                if task_user_id and user_id != task_user_id:
                    continue

                # Check if task is stale (running for >10 minutes without completion)
                # Auto-complete it to prevent infinite polling loops
                created_at_str = meta.get("created_at")
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                        age_seconds = (
                            datetime.now(timezone.utc) - created_at
                        ).total_seconds()
                        if age_seconds > 600:  # 10 minutes
                            logger.warning(
                                f"[STALE_TASK] Auto-completing stale session {session_id[:8]}... "
                                f"(running for {age_seconds:.0f}s)"
                            )
                            await mark_task_completed(
                                session_id,
                                status="failed",
                                error_message=f"Task timed out after {age_seconds:.0f}s",
                            )
                            continue  # Skip this task, it's now completed
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse created_at: {e}")

                logger.info(f"[TASK_LOOKUP] Found running session {session_id[:8]}...")

                # Get the last message ID from Redis Stream
                stream_key = _get_task_stream_key(session_id)
                last_id = "0-0"
                try:
                    messages = await redis.xrevrange(stream_key, count=1)
                    if messages:
                        msg_id = messages[0][0]
                        last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                except Exception as e:
                    logger.warning(f"Failed to get last message ID: {e}")

                return (
                    ActiveTask(
                        session_id=session_id,
                        user_id=task_user_id,
                        tool_call_id=meta.get("tool_call_id", ""),
                        tool_name=meta.get("tool_name", ""),
                        operation_id=meta.get("operation_id", ""),
                        status="running",
                    ),
                    last_id,
                )

        if cursor == 0:
            break

    return None, "0-0"


def _reconstruct_chunk(chunk_data: dict) -> StreamBaseResponse | None:
    """Reconstruct a StreamBaseResponse from JSON data.

    Args:
        chunk_data: Parsed JSON data from Redis

    Returns:
        Reconstructed response object, or None if unknown type
    """
    from .response_model import (
        ResponseType,
        StreamError,
        StreamFinish,
        StreamFinishStep,
        StreamHeartbeat,
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

    # Map response types to their corresponding classes
    type_to_class: dict[str, type[StreamBaseResponse]] = {
        ResponseType.START.value: StreamStart,
        ResponseType.FINISH.value: StreamFinish,
        ResponseType.START_STEP.value: StreamStartStep,
        ResponseType.FINISH_STEP.value: StreamFinishStep,
        ResponseType.TEXT_START.value: StreamTextStart,
        ResponseType.TEXT_DELTA.value: StreamTextDelta,
        ResponseType.TEXT_END.value: StreamTextEnd,
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


async def set_task_asyncio_task(session_id: str, asyncio_task: asyncio.Task) -> None:
    """Track the asyncio.Task for a task (local reference only).

    This is just for cleanup purposes - the task state is in Redis.

    Args:
        session_id: Session ID
        asyncio_task: The asyncio Task to track
    """
    _local_tasks[session_id] = asyncio_task


async def unsubscribe_from_task(
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
    listener_entry = _listener_tasks.pop(queue_id, None)

    if listener_entry is None:
        logger.debug(
            f"No listener task found for task {session_id} queue {queue_id} "
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
        logger.debug(f"Listener task for task {session_id} already completed")
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
            f"Timeout waiting for listener task cancellation for task {session_id}"
        )
    except Exception as e:
        logger.error(
            f"Error during listener task cancellation for task {session_id}: {e}"
        )

    logger.debug(f"Successfully unsubscribed from task {session_id}")
