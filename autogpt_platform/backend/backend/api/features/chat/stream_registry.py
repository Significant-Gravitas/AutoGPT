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
from .response_model import StreamBaseResponse, StreamError, StreamFinish

logger = logging.getLogger(__name__)
config = ChatConfig()

# Track background tasks for this pod (just the asyncio.Task reference, not subscribers)
_local_tasks: dict[str, asyncio.Task] = {}

# Track listener tasks per subscriber queue for cleanup
# Maps queue id() to (task_id, asyncio.Task) for proper cleanup on unsubscribe
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

    task_id: str
    session_id: str
    user_id: str | None
    tool_call_id: str
    tool_name: str
    operation_id: str
    status: Literal["running", "completed", "failed"] = "running"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    asyncio_task: asyncio.Task | None = None


def _get_task_meta_key(task_id: str) -> str:
    """Get Redis key for task metadata."""
    return f"{config.task_meta_prefix}{task_id}"


def _get_task_stream_key(task_id: str) -> str:
    """Get Redis key for task message stream."""
    return f"{config.task_stream_prefix}{task_id}"


def _get_operation_mapping_key(operation_id: str) -> str:
    """Get Redis key for operation_id to task_id mapping."""
    return f"{config.task_op_prefix}{operation_id}"


async def create_task(
    task_id: str,
    session_id: str,
    user_id: str | None,
    tool_call_id: str,
    tool_name: str,
    operation_id: str,
) -> ActiveTask:
    """Create a new streaming task in Redis.

    Args:
        task_id: Unique identifier for the task
        session_id: Chat session ID
        user_id: User ID (may be None for anonymous)
        tool_call_id: Tool call ID from the LLM
        tool_name: Name of the tool being executed
        operation_id: Operation ID for webhook callbacks

    Returns:
        The created ActiveTask instance (metadata only)
    """
    task = ActiveTask(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        operation_id=operation_id,
    )

    # Store metadata in Redis
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    op_key = _get_operation_mapping_key(operation_id)

    await redis.hset(  # type: ignore[misc]
        meta_key,
        mapping={
            "task_id": task_id,
            "session_id": session_id,
            "user_id": user_id or "",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "operation_id": operation_id,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
        },
    )
    await redis.expire(meta_key, config.stream_ttl)

    # Create operation_id -> task_id mapping for webhook lookups
    await redis.set(op_key, task_id, ex=config.stream_ttl)

    logger.debug(f"Created task {task_id} for session {session_id}")

    return task


async def publish_chunk(
    task_id: str,
    chunk: StreamBaseResponse,
) -> str:
    """Publish a chunk to Redis Stream.

    All delivery is via Redis Streams - no in-memory state.

    Args:
        task_id: Task ID to publish to
        chunk: The stream response chunk to publish

    Returns:
        The Redis Stream message ID
    """
    chunk_json = chunk.model_dump_json()
    message_id = "0-0"

    try:
        redis = await get_redis_async()
        stream_key = _get_task_stream_key(task_id)

        # Write to Redis Stream for persistence and real-time delivery
        raw_id = await redis.xadd(
            stream_key,
            {"data": chunk_json},
            maxlen=config.stream_max_length,
        )
        message_id = raw_id if isinstance(raw_id, str) else raw_id.decode()

        # Set TTL on stream to match task metadata TTL
        await redis.expire(stream_key, config.stream_ttl)
    except Exception as e:
        logger.error(
            f"Failed to publish chunk for task {task_id}: {e}",
            exc_info=True,
        )

    return message_id


async def subscribe_to_task(
    task_id: str,
    user_id: str | None,
    last_message_id: str = "0-0",
) -> asyncio.Queue[StreamBaseResponse] | None:
    """Subscribe to a task's stream with replay of missed messages.

    This is fully stateless - uses Redis Stream for replay and pub/sub for live updates.

    Args:
        task_id: Task ID to subscribe to
        user_id: User ID for ownership validation
        last_message_id: Last Redis Stream message ID received ("0-0" for full replay)

    Returns:
        An asyncio Queue that will receive stream chunks, or None if task not found
        or user doesn't have access
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        logger.debug(f"Task {task_id} not found in Redis")
        return None

    # Note: Redis client uses decode_responses=True, so keys are strings
    task_status = meta.get("status", "")
    task_user_id = meta.get("user_id", "") or None

    # Validate ownership - if task has an owner, requester must match
    if task_user_id:
        if user_id != task_user_id:
            logger.warning(
                f"User {user_id} denied access to task {task_id} "
                f"owned by {task_user_id}"
            )
            return None

    subscriber_queue: asyncio.Queue[StreamBaseResponse] = asyncio.Queue()
    stream_key = _get_task_stream_key(task_id)

    # Step 1: Replay messages from Redis Stream
    messages = await redis.xread({stream_key: last_message_id}, block=0, count=1000)

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

    logger.debug(f"Task {task_id}: replayed {replayed_count} messages")

    # Step 2: If task is still running, start stream listener for live updates
    if task_status == "running":
        listener_task = asyncio.create_task(
            _stream_listener(task_id, subscriber_queue, replay_last_id)
        )
        # Track listener task for cleanup on unsubscribe
        _listener_tasks[id(subscriber_queue)] = (task_id, listener_task)
    else:
        # Task is completed/failed - add finish marker
        await subscriber_queue.put(StreamFinish())

    return subscriber_queue


async def _stream_listener(
    task_id: str,
    subscriber_queue: asyncio.Queue[StreamBaseResponse],
    last_replayed_id: str,
) -> None:
    """Listen to Redis Stream for new messages using blocking XREAD.

    This approach avoids the duplicate message issue that can occur with pub/sub
    when messages are published during the gap between replay and subscription.

    Args:
        task_id: Task ID to listen for
        subscriber_queue: Queue to deliver messages to
        last_replayed_id: Last message ID from replay (continue from here)
    """
    queue_id = id(subscriber_queue)
    # Track the last successfully delivered message ID for recovery hints
    last_delivered_id = last_replayed_id

    try:
        redis = await get_redis_async()
        stream_key = _get_task_stream_key(task_id)
        current_id = last_replayed_id

        while True:
            # Block for up to 30 seconds waiting for new messages
            # This allows periodic checking if task is still running
            messages = await redis.xread(
                {stream_key: current_id}, block=30000, count=100
            )

            if not messages:
                # Timeout - check if task is still running
                meta_key = _get_task_meta_key(task_id)
                status = await redis.hget(meta_key, "status")  # type: ignore[misc]
                if status and status != "running":
                    try:
                        await asyncio.wait_for(
                            subscriber_queue.put(StreamFinish()),
                            timeout=QUEUE_PUT_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout delivering finish event for task {task_id}"
                        )
                    break
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
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"Subscriber queue full for task {task_id}, "
                                    f"message delivery timed out after {QUEUE_PUT_TIMEOUT}s"
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
                                        f"Cannot deliver overflow error for task {task_id}, "
                                        "queue completely blocked"
                                    )

                            # Stop listening on finish
                            if isinstance(chunk, StreamFinish):
                                return
                    except Exception as e:
                        logger.warning(f"Error processing stream message: {e}")

    except asyncio.CancelledError:
        logger.debug(f"Stream listener cancelled for task {task_id}")
        raise  # Re-raise to propagate cancellation
    except Exception as e:
        logger.error(f"Stream listener error for task {task_id}: {e}")
        # On error, send finish to unblock subscriber
        try:
            await asyncio.wait_for(
                subscriber_queue.put(StreamFinish()),
                timeout=QUEUE_PUT_TIMEOUT,
            )
        except (asyncio.TimeoutError, asyncio.QueueFull):
            logger.warning(
                f"Could not deliver finish event for task {task_id} after error"
            )
    finally:
        # Clean up listener task mapping on exit
        _listener_tasks.pop(queue_id, None)


async def mark_task_completed(
    task_id: str,
    status: Literal["completed", "failed"] = "completed",
) -> bool:
    """Mark a task as completed and publish finish event.

    This is idempotent - calling multiple times with the same task_id is safe.
    Uses atomic compare-and-swap via Lua script to prevent race conditions.
    Status is updated first (source of truth), then finish event is published (best-effort).

    Args:
        task_id: Task ID to mark as completed
        status: Final status ("completed" or "failed")

    Returns:
        True if task was newly marked completed, False if already completed/failed
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)

    # Atomic compare-and-swap: only update if status is "running"
    # This prevents race conditions when multiple callers try to complete simultaneously
    result = await redis.eval(COMPLETE_TASK_SCRIPT, 1, meta_key, status)  # type: ignore[misc]

    if result == 0:
        logger.debug(f"Task {task_id} already completed/failed, skipping")
        return False

    # THEN publish finish event (best-effort - listeners can detect via status polling)
    try:
        await publish_chunk(task_id, StreamFinish())
    except Exception as e:
        logger.error(
            f"Failed to publish finish event for task {task_id}: {e}. "
            "Listeners will detect completion via status polling."
        )

    # Clean up local task reference if exists
    _local_tasks.pop(task_id, None)
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
    task_id = await redis.get(op_key)

    if not task_id:
        return None

    task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
    return await get_task(task_id_str)


async def get_task(task_id: str) -> ActiveTask | None:
    """Get a task by its ID from Redis.

    Args:
        task_id: Task ID to look up

    Returns:
        ActiveTask if found, None otherwise
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        return None

    # Note: Redis client uses decode_responses=True, so keys/values are strings
    return ActiveTask(
        task_id=meta.get("task_id", ""),
        session_id=meta.get("session_id", ""),
        user_id=meta.get("user_id", "") or None,
        tool_call_id=meta.get("tool_call_id", ""),
        tool_name=meta.get("tool_name", ""),
        operation_id=meta.get("operation_id", ""),
        status=meta.get("status", "running"),  # type: ignore[arg-type]
    )


async def get_task_with_expiry_info(
    task_id: str,
) -> tuple[ActiveTask | None, str | None]:
    """Get a task by its ID with expiration detection.

    Returns (task, error_code) where error_code is:
    - None if task found
    - "TASK_EXPIRED" if stream exists but metadata is gone (TTL expired)
    - "TASK_NOT_FOUND" if neither exists

    Args:
        task_id: Task ID to look up

    Returns:
        Tuple of (ActiveTask or None, error_code or None)
    """
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    stream_key = _get_task_stream_key(task_id)

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
            task_id=meta.get("task_id", ""),
            session_id=meta.get("session_id", ""),
            user_id=meta.get("user_id", "") or None,
            tool_call_id=meta.get("tool_call_id", ""),
            tool_name=meta.get("tool_name", ""),
            operation_id=meta.get("operation_id", ""),
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
            task_id = meta.get("task_id", "")

            if task_session_id == session_id and task_status == "running":
                # Validate ownership - if task has an owner, requester must match
                if task_user_id and user_id != task_user_id:
                    continue

                # Get the last message ID from Redis Stream
                stream_key = _get_task_stream_key(task_id)
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
                        task_id=task_id,
                        session_id=task_session_id,
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
        StreamHeartbeat,
        StreamStart,
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


async def set_task_asyncio_task(task_id: str, asyncio_task: asyncio.Task) -> None:
    """Track the asyncio.Task for a task (local reference only).

    This is just for cleanup purposes - the task state is in Redis.

    Args:
        task_id: Task ID
        asyncio_task: The asyncio Task to track
    """
    _local_tasks[task_id] = asyncio_task


async def unsubscribe_from_task(
    task_id: str,
    subscriber_queue: asyncio.Queue[StreamBaseResponse],
) -> None:
    """Clean up when a subscriber disconnects.

    Cancels the XREAD-based listener task associated with this subscriber queue
    to prevent resource leaks.

    Args:
        task_id: Task ID
        subscriber_queue: The subscriber's queue used to look up the listener task
    """
    queue_id = id(subscriber_queue)
    listener_entry = _listener_tasks.pop(queue_id, None)

    if listener_entry is None:
        logger.debug(
            f"No listener task found for task {task_id} queue {queue_id} "
            "(may have already completed)"
        )
        return

    stored_task_id, listener_task = listener_entry

    if stored_task_id != task_id:
        logger.warning(
            f"Task ID mismatch in unsubscribe: expected {task_id}, "
            f"found {stored_task_id}"
        )

    if listener_task.done():
        logger.debug(f"Listener task for task {task_id} already completed")
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
            f"Timeout waiting for listener task cancellation for task {task_id}"
        )
    except Exception as e:
        logger.error(f"Error during listener task cancellation for task {task_id}: {e}")

    logger.debug(f"Successfully unsubscribed from task {task_id}")
