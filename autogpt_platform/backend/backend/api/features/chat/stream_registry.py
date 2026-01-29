"""Stream registry for managing reconnectable SSE streams.

This module provides a registry for tracking active streaming tasks and their
messages. It supports:
- Creating tasks with unique IDs for long-running operations
- Publishing stream messages to both Redis Streams and in-memory queues
- Subscribing to tasks with replay of missed messages
- Looking up tasks by operation_id for webhook callbacks
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
    """Represents an active streaming task."""

    task_id: str
    session_id: str
    user_id: str | None
    tool_call_id: str
    tool_name: str
    operation_id: str
    status: Literal["running", "completed", "failed"] = "running"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    queue: asyncio.Queue[StreamBaseResponse] = field(default_factory=asyncio.Queue)
    asyncio_task: asyncio.Task | None = None


# Module-level registry for active tasks
_active_tasks: dict[str, ActiveTask] = {}

# Redis key patterns
TASK_META_PREFIX = "chat:task:meta:"  # Hash for task metadata
TASK_STREAM_PREFIX = "chat:stream:"  # Redis Stream for messages
TASK_OP_PREFIX = "chat:task:op:"  # Operation ID -> task_id mapping


def _get_task_meta_key(task_id: str) -> str:
    """Get Redis key for task metadata."""
    return f"{TASK_META_PREFIX}{task_id}"


def _get_task_stream_key(task_id: str) -> str:
    """Get Redis key for task message stream."""
    return f"{TASK_STREAM_PREFIX}{task_id}"


def _get_operation_mapping_key(operation_id: str) -> str:
    """Get Redis key for operation_id to task_id mapping."""
    return f"{TASK_OP_PREFIX}{operation_id}"


async def create_task(
    task_id: str,
    session_id: str,
    user_id: str | None,
    tool_call_id: str,
    tool_name: str,
    operation_id: str,
) -> ActiveTask:
    """Create a new streaming task in memory and Redis.

    Args:
        task_id: Unique identifier for the task
        session_id: Chat session ID
        user_id: User ID (may be None for anonymous)
        tool_call_id: Tool call ID from the LLM
        tool_name: Name of the tool being executed
        operation_id: Operation ID for webhook callbacks

    Returns:
        The created ActiveTask instance
    """
    task = ActiveTask(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        operation_id=operation_id,
    )

    # Store in memory registry
    _active_tasks[task_id] = task

    # Store metadata in Redis for durability
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
        f"Created streaming task {task_id} for operation {operation_id} "
        f"in session {session_id}"
    )

    return task


async def publish_chunk(
    task_id: str,
    chunk: StreamBaseResponse,
) -> int:
    """Publish a chunk to the task's stream.

    Writes to both Redis Stream (for replay) and in-memory queue (for live subscribers).

    Args:
        task_id: Task ID to publish to
        chunk: The stream response chunk to publish

    Returns:
        The message index in the Redis Stream
    """
    redis = await get_redis_async()
    stream_key = _get_task_stream_key(task_id)

    # Serialize chunk to JSON
    chunk_json = chunk.model_dump_json()

    # Add to Redis Stream with auto-generated ID
    # The ID format is "timestamp-sequence" which gives us ordering
    message_id = await redis.xadd(
        stream_key,
        {"data": chunk_json},
        maxlen=config.stream_max_length,
    )

    # Publish to in-memory queue if task exists
    task = _active_tasks.get(task_id)
    if task:
        try:
            task.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.warning(f"Queue full for task {task_id}, dropping chunk")

    logger.debug(f"Published chunk to task {task_id}, message_id={message_id}")

    # Parse the message_id to extract the index
    # Redis Stream IDs are "timestamp-sequence", we return the raw ID
    return int(message_id.split("-")[1]) if "-" in message_id else 0


async def subscribe_to_task(
    task_id: str,
    user_id: str | None,
    last_idx: int = 0,
) -> asyncio.Queue[StreamBaseResponse] | None:
    """Subscribe to a task's stream with replay of missed messages.

    Args:
        task_id: Task ID to subscribe to
        user_id: User ID for ownership validation
        last_idx: Last message index received (0 for full replay)

    Returns:
        An asyncio Queue that will receive stream chunks, or None if task not found
        or user doesn't have access
    """
    # Check in-memory first
    task = _active_tasks.get(task_id)

    if task:
        # Validate ownership
        if user_id and task.user_id and task.user_id != user_id:
            logger.warning(
                f"User {user_id} attempted to subscribe to task {task_id} "
                f"owned by {task.user_id}"
            )
            return None

        # Create a new queue for this subscriber
        subscriber_queue: asyncio.Queue[StreamBaseResponse] = asyncio.Queue()

        # Replay from Redis Stream
        redis = await get_redis_async()
        stream_key = _get_task_stream_key(task_id)

        # Read all messages from stream
        # Use "0-0" to get all messages or construct ID from last_idx
        start_id = "0-0" if last_idx == 0 else f"0-{last_idx}"
        messages = await redis.xread({stream_key: start_id}, block=0, count=1000)

        if messages:
            # messages format: [[stream_name, [(id, {data: json}), ...]]]
            for _stream_name, stream_messages in messages:
                for _msg_id, msg_data in stream_messages:
                    if b"data" in msg_data:
                        try:
                            chunk_data = orjson.loads(msg_data[b"data"])
                            # Reconstruct the appropriate response type
                            chunk = _reconstruct_chunk(chunk_data)
                            if chunk:
                                await subscriber_queue.put(chunk)
                        except Exception as e:
                            logger.warning(f"Failed to replay message: {e}")

        # If task is still running, set up live subscription
        if task.status == "running":
            # Forward messages from task queue to subscriber queue
            async def _forward_messages():
                try:
                    while True:
                        chunk = await task.queue.get()
                        await subscriber_queue.put(chunk)
                        if isinstance(chunk, StreamFinish):
                            break
                except asyncio.CancelledError:
                    pass

            asyncio.create_task(_forward_messages())
        else:
            # Task is done, add finish marker
            await subscriber_queue.put(StreamFinish())

        return subscriber_queue

    # Try to load from Redis if not in memory
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if not meta:
        logger.warning(f"Task {task_id} not found in memory or Redis")
        return None

    # Validate ownership
    task_user_id = meta.get(b"user_id", b"").decode() or None
    if user_id and task_user_id and task_user_id != user_id:
        logger.warning(
            f"User {user_id} attempted to subscribe to task {task_id} "
            f"owned by {task_user_id}"
        )
        return None

    # Replay from Redis Stream only (task is not in memory, so it's completed/crashed)
    subscriber_queue = asyncio.Queue()
    stream_key = _get_task_stream_key(task_id)

    start_id = "0-0" if last_idx == 0 else f"0-{last_idx}"
    messages = await redis.xread({stream_key: start_id}, block=0, count=1000)

    if messages:
        for _stream_name, stream_messages in messages:
            for _msg_id, msg_data in stream_messages:
                if b"data" in msg_data:
                    try:
                        chunk_data = orjson.loads(msg_data[b"data"])
                        chunk = _reconstruct_chunk(chunk_data)
                        if chunk:
                            await subscriber_queue.put(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to replay message: {e}")

    # Add finish marker since task is not active
    await subscriber_queue.put(StreamFinish())

    return subscriber_queue


async def mark_task_completed(
    task_id: str,
    status: Literal["completed", "failed"] = "completed",
) -> None:
    """Mark a task as completed and publish final event.

    Args:
        task_id: Task ID to mark as completed
        status: Final status ("completed" or "failed")
    """
    task = _active_tasks.get(task_id)

    if task:
        task.status = status
        # Publish finish event to all subscribers
        await publish_chunk(task_id, StreamFinish())

        # Remove from active tasks after a short delay to allow subscribers to finish
        async def _cleanup():
            await asyncio.sleep(5)
            _active_tasks.pop(task_id, None)
            logger.info(f"Cleaned up task {task_id} from memory")

        asyncio.create_task(_cleanup())

    # Update Redis metadata
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    await redis.hset(meta_key, "status", status)  # type: ignore[misc]

    logger.info(f"Marked task {task_id} as {status}")


async def find_task_by_operation_id(operation_id: str) -> ActiveTask | None:
    """Find a task by its operation ID.

    Used by webhook callbacks to locate the task to update.

    Args:
        operation_id: Operation ID to search for

    Returns:
        ActiveTask if found, None otherwise
    """
    # Check in-memory first
    for task in _active_tasks.values():
        if task.operation_id == operation_id:
            return task

    # Try Redis lookup
    redis = await get_redis_async()
    op_key = _get_operation_mapping_key(operation_id)
    task_id = await redis.get(op_key)

    if task_id:
        task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
        # Check if task is in memory
        if task_id_str in _active_tasks:
            return _active_tasks[task_id_str]

        # Load metadata from Redis
        meta_key = _get_task_meta_key(task_id_str)
        meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

        if meta:
            # Reconstruct task object (not fully active, but has metadata)
            return ActiveTask(
                task_id=meta.get(b"task_id", b"").decode(),
                session_id=meta.get(b"session_id", b"").decode(),
                user_id=meta.get(b"user_id", b"").decode() or None,
                tool_call_id=meta.get(b"tool_call_id", b"").decode(),
                tool_name=meta.get(b"tool_name", b"").decode(),
                operation_id=operation_id,
                status=meta.get(b"status", b"running").decode(),  # type: ignore
            )

    return None


async def get_task(task_id: str) -> ActiveTask | None:
    """Get a task by its ID.

    Args:
        task_id: Task ID to look up

    Returns:
        ActiveTask if found, None otherwise
    """
    # Check in-memory first
    if task_id in _active_tasks:
        return _active_tasks[task_id]

    # Try Redis lookup
    redis = await get_redis_async()
    meta_key = _get_task_meta_key(task_id)
    meta: dict[Any, Any] = await redis.hgetall(meta_key)  # type: ignore[misc]

    if meta:
        return ActiveTask(
            task_id=meta.get(b"task_id", b"").decode(),
            session_id=meta.get(b"session_id", b"").decode(),
            user_id=meta.get(b"user_id", b"").decode() or None,
            tool_call_id=meta.get(b"tool_call_id", b"").decode(),
            tool_name=meta.get(b"tool_name", b"").decode(),
            operation_id=meta.get(b"operation_id", b"").decode(),
            status=meta.get(b"status", b"running").decode(),  # type: ignore[arg-type]
        )

    return None


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
    """Associate an asyncio.Task with an ActiveTask.

    Args:
        task_id: Task ID
        asyncio_task: The asyncio Task to associate
    """
    task = _active_tasks.get(task_id)
    if task:
        task.asyncio_task = asyncio_task
