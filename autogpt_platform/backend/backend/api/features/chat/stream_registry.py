"""Stream registry for managing reconnectable SSE streams.

This module provides a registry for tracking active streaming tasks and their
messages. It uses Redis for all state management (no in-memory state), making
pods stateless and horizontally scalable.

Architecture:
- Redis Stream: Persists all messages for replay
- Redis Pub/Sub: Real-time delivery to subscribers
- Redis Hash: Task metadata (status, session_id, etc.)

Subscribers:
1. Replay missed messages from Redis Stream
2. Subscribe to pub/sub channel for live updates
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
from .response_model import StreamBaseResponse, StreamFinish

logger = logging.getLogger(__name__)
config = ChatConfig()


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


# Redis key patterns
TASK_META_PREFIX = "chat:task:meta:"  # Hash for task metadata
TASK_STREAM_PREFIX = "chat:stream:"  # Redis Stream for messages
TASK_OP_PREFIX = "chat:task:op:"  # Operation ID -> task_id mapping
TASK_PUBSUB_PREFIX = "chat:task:pubsub:"  # Pub/sub channel for real-time delivery

# Track background tasks for this pod (just the asyncio.Task reference, not subscribers)
_local_tasks: dict[str, asyncio.Task] = {}


def _get_task_meta_key(task_id: str) -> str:
    """Get Redis key for task metadata."""
    return f"{TASK_META_PREFIX}{task_id}"


def _get_task_stream_key(task_id: str) -> str:
    """Get Redis key for task message stream."""
    return f"{TASK_STREAM_PREFIX}{task_id}"


def _get_operation_mapping_key(operation_id: str) -> str:
    """Get Redis key for operation_id to task_id mapping."""
    return f"{TASK_OP_PREFIX}{operation_id}"


def _get_task_pubsub_channel(task_id: str) -> str:
    """Get Redis pub/sub channel for task real-time delivery."""
    return f"{TASK_PUBSUB_PREFIX}{task_id}"


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

    logger.info(
        f"[SSE-RECONNECT] Created task {task_id} for session {session_id} in Redis"
    )

    return task


async def publish_chunk(
    task_id: str,
    chunk: StreamBaseResponse,
) -> str:
    """Publish a chunk to Redis Stream and pub/sub channel.

    All delivery is via Redis - no in-memory state.

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
        pubsub_channel = _get_task_pubsub_channel(task_id)

        # Write to Redis Stream for persistence/replay
        raw_id = await redis.xadd(
            stream_key,
            {"data": chunk_json},
            maxlen=config.stream_max_length,
        )
        message_id = raw_id if isinstance(raw_id, str) else raw_id.decode()

        # Publish to pub/sub for real-time delivery
        await redis.publish(pubsub_channel, chunk_json)

        logger.debug(f"Published chunk to task {task_id}, message_id={message_id}")
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
        logger.warning(f"[SSE-RECONNECT] Task {task_id} not found in Redis")
        return None

    # Note: Redis client uses decode_responses=True, so keys are strings
    task_status = meta.get("status", "")
    task_user_id = meta.get("user_id", "") or None

    logger.info(f"[SSE-RECONNECT] Subscribing to task {task_id}: status={task_status}")

    # Validate ownership
    if user_id and task_user_id and task_user_id != user_id:
        logger.warning(
            f"User {user_id} attempted to subscribe to task {task_id} "
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

    logger.info(
        f"[SSE-RECONNECT] Task {task_id}: replayed {replayed_count} messages "
        f"(last_id={replay_last_id})"
    )

    # Step 2: If task is still running, start stream listener for live updates
    if task_status == "running":
        logger.info(
            f"[SSE-RECONNECT] Task {task_id} is running, starting stream listener"
        )
        asyncio.create_task(_stream_listener(task_id, subscriber_queue, replay_last_id))
    else:
        # Task is completed/failed - add finish marker
        logger.info(
            f"[SSE-RECONNECT] Task {task_id} is {task_status}, adding finish marker"
        )
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
    try:
        redis = await get_redis_async()
        stream_key = _get_task_stream_key(task_id)
        current_id = last_replayed_id

        logger.debug(
            f"[SSE-RECONNECT] Stream listener started for task {task_id}, "
            f"from ID {current_id}"
        )

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
                    logger.info(
                        f"[SSE-RECONNECT] Task {task_id} no longer running "
                        f"(status={status}), stopping listener"
                    )
                    subscriber_queue.put_nowait(StreamFinish())
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
                                subscriber_queue.put_nowait(chunk)
                            except asyncio.QueueFull:
                                logger.warning(
                                    f"Subscriber queue full for task {task_id}"
                                )

                            # Stop listening on finish
                            if isinstance(chunk, StreamFinish):
                                logger.info(
                                    f"[SSE-RECONNECT] Task {task_id} finished "
                                    "via stream"
                                )
                                return
                    except Exception as e:
                        logger.warning(f"Error processing stream message: {e}")

    except asyncio.CancelledError:
        logger.debug(f"[SSE-RECONNECT] Stream listener cancelled for task {task_id}")
    except Exception as e:
        logger.error(f"Stream listener error for task {task_id}: {e}")
        # On error, send finish to unblock subscriber
        try:
            subscriber_queue.put_nowait(StreamFinish())
        except asyncio.QueueFull:
            pass


async def mark_task_completed(
    task_id: str,
    status: Literal["completed", "failed"] = "completed",
) -> None:
    """Mark a task as completed and publish finish event.

    Args:
        task_id: Task ID to mark as completed
        status: Final status ("completed" or "failed")
    """
    # Publish finish event (goes to Redis Stream + pub/sub)
    await publish_chunk(task_id, StreamFinish())

    # Update Redis metadata
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    await redis.hset(meta_key, "status", status)  # type: ignore[misc]

    # Clean up local task reference if exists
    _local_tasks.pop(task_id, None)

    logger.info(f"[SSE-RECONNECT] Marked task {task_id} as {status}")


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

    logger.info(
        f"[SSE-RECONNECT] find_task_by_operation_id: "
        f"op_key={op_key}, task_id_from_redis={task_id!r}"
    )

    if not task_id:
        logger.info(f"[SSE-RECONNECT] No task_id found for operation {operation_id}")
        return None

    task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
    logger.info(f"[SSE-RECONNECT] Looking up task by task_id={task_id_str}")
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

    logger.info(
        f"[SSE-RECONNECT] get_task: meta_key={meta_key}, "
        f"meta_keys={list(meta.keys()) if meta else 'empty'}, "
        f"meta={meta}"
    )

    if not meta:
        logger.info(f"[SSE-RECONNECT] No metadata found for task {task_id}")
        return None

    # Note: Redis client uses decode_responses=True, so keys/values are strings
    task = ActiveTask(
        task_id=meta.get("task_id", ""),
        session_id=meta.get("session_id", ""),
        user_id=meta.get("user_id", "") or None,
        tool_call_id=meta.get("tool_call_id", ""),
        tool_name=meta.get("tool_name", ""),
        operation_id=meta.get("operation_id", ""),
        status=meta.get("status", "running"),  # type: ignore[arg-type]
    )
    logger.info(
        f"[SSE-RECONNECT] get_task returning: task_id={task.task_id}, "
        f"session_id={task.session_id}, operation_id={task.operation_id}"
    )
    return task


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
    logger.info(f"[SSE-RECONNECT] Looking for active task for session {session_id}")

    redis = await get_redis_async()

    # Scan Redis for task metadata keys
    cursor = 0
    tasks_checked = 0

    while True:
        cursor, keys = await redis.scan(cursor, match=f"{TASK_META_PREFIX}*", count=100)

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

            # Log tasks found for this session
            if task_session_id == session_id:
                logger.info(
                    f"[SSE-RECONNECT] Found task for session: "
                    f"task_id={task_id}, status={task_status}"
                )

            if task_session_id == session_id and task_status == "running":
                # Validate ownership
                if user_id and task_user_id and task_user_id != user_id:
                    logger.info(f"[SSE-RECONNECT] Task {task_id} ownership mismatch")
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

                logger.info(
                    f"[SSE-RECONNECT] Found active task: task_id={task_id}, "
                    f"last_message_id={last_id}"
                )

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

    logger.info(
        f"[SSE-RECONNECT] No active task found for session {session_id} "
        f"(checked {tasks_checked} tasks)"
    )
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

    chunk_type = chunk_data.get("type")

    try:
        if chunk_type == ResponseType.START.value:
            return StreamStart(**chunk_data)
        elif chunk_type == ResponseType.FINISH.value:
            return StreamFinish(**chunk_data)
        elif chunk_type == ResponseType.TEXT_START.value:
            return StreamTextStart(**chunk_data)
        elif chunk_type == ResponseType.TEXT_DELTA.value:
            return StreamTextDelta(**chunk_data)
        elif chunk_type == ResponseType.TEXT_END.value:
            return StreamTextEnd(**chunk_data)
        elif chunk_type == ResponseType.TOOL_INPUT_START.value:
            return StreamToolInputStart(**chunk_data)
        elif chunk_type == ResponseType.TOOL_INPUT_AVAILABLE.value:
            return StreamToolInputAvailable(**chunk_data)
        elif chunk_type == ResponseType.TOOL_OUTPUT_AVAILABLE.value:
            return StreamToolOutputAvailable(**chunk_data)
        elif chunk_type == ResponseType.ERROR.value:
            return StreamError(**chunk_data)
        elif chunk_type == ResponseType.USAGE.value:
            return StreamUsage(**chunk_data)
        elif chunk_type == ResponseType.HEARTBEAT.value:
            return StreamHeartbeat(**chunk_data)
        else:
            logger.warning(f"Unknown chunk type: {chunk_type}")
            return None
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

    With Redis-based pub/sub, there's no explicit unsubscription needed.
    The pub/sub listener task will be garbage collected when the subscriber
    stops reading from the queue.

    Args:
        task_id: Task ID
        subscriber_queue: The subscriber's queue (unused, kept for API compat)
    """
    # No-op - pub/sub listener cleans up automatically
    logger.debug(f"[SSE-RECONNECT] Subscriber disconnected from task {task_id}")
