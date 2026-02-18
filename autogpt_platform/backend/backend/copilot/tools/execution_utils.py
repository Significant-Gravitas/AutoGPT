"""Shared utilities for execution waiting and status handling."""

import asyncio
import logging
from typing import Any

from backend.data.db_accessors import execution_db
from backend.data.execution import (
    AsyncRedisExecutionEventBus,
    ExecutionStatus,
    GraphExecution,
    GraphExecutionEvent,
)

logger = logging.getLogger(__name__)

# Terminal statuses that indicate execution is complete
TERMINAL_STATUSES = frozenset(
    {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.TERMINATED,
    }
)

# Statuses where execution is paused but not finished (e.g. human-in-the-loop)
PAUSED_STATUSES = frozenset(
    {
        ExecutionStatus.REVIEW,
    }
)

# Statuses that mean "stop waiting" (terminal or paused)
STOP_WAITING_STATUSES = TERMINAL_STATUSES | PAUSED_STATUSES


async def wait_for_execution(
    user_id: str,
    graph_id: str,
    execution_id: str,
    timeout_seconds: int,
) -> GraphExecution | None:
    """
    Wait for an execution to reach a terminal or paused status using Redis pubsub.

    Handles the race condition between checking status and subscribing by
    re-checking the DB after the subscription is established.

    Args:
        user_id: User ID
        graph_id: Graph ID
        execution_id: Execution ID to wait for
        timeout_seconds: Max seconds to wait

    Returns:
        The execution with current status, or None if not found
    """
    exec_db = execution_db()

    # Quick check — maybe it's already done
    execution = await exec_db.get_graph_execution(
        user_id=user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )
    if not execution:
        return None

    if execution.status in STOP_WAITING_STATUSES:
        logger.debug(
            f"Execution {execution_id} already in stop-waiting state: "
            f"{execution.status}"
        )
        return execution

    logger.info(
        f"Waiting up to {timeout_seconds}s for execution {execution_id} "
        f"(current status: {execution.status})"
    )

    event_bus = AsyncRedisExecutionEventBus()
    channel_key = f"{user_id}/{graph_id}/{execution_id}"

    try:
        result = await asyncio.wait_for(
            _subscribe_and_wait(
                event_bus, channel_key, user_id, execution_id, exec_db
            ),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        logger.info(f"Timeout waiting for execution {execution_id}")
    except Exception as e:
        logger.error(f"Error waiting for execution: {e}", exc_info=True)
    finally:
        await event_bus.close()

    # Return current state on timeout/error
    return await exec_db.get_graph_execution(
        user_id=user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )


async def _subscribe_and_wait(
    event_bus: AsyncRedisExecutionEventBus,
    channel_key: str,
    user_id: str,
    execution_id: str,
    exec_db: Any,
) -> GraphExecution | None:
    """
    Subscribe to execution events and wait for a terminal/paused status.

    To avoid the race condition where the execution completes between the
    initial DB check and the Redis subscription, we:
    1. Start listening (which subscribes internally)
    2. Re-check the DB after subscription is active
    3. If still running, wait for pubsub events
    """
    listen_iter = event_bus.listen_events(channel_key).__aiter__()

    # Start a background task to consume pubsub events
    result_event: GraphExecutionEvent | None = None
    done = asyncio.Event()

    async def _consume():
        nonlocal result_event
        async for event in listen_iter:
            if isinstance(event, GraphExecutionEvent):
                logger.debug(f"Received execution update: {event.status}")
                if event.status in STOP_WAITING_STATUSES:
                    result_event = event
                    done.set()
                    return

    consume_task = asyncio.create_task(_consume())

    # Give the subscription a moment to establish, then re-check DB
    await asyncio.sleep(0.1)

    execution = await exec_db.get_graph_execution(
        user_id=user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )
    if execution and execution.status in STOP_WAITING_STATUSES:
        consume_task.cancel()
        return execution

    # Wait for the pubsub consumer to find a terminal event
    await done.wait()
    consume_task.cancel()

    # Fetch full execution
    return await exec_db.get_graph_execution(
        user_id=user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )


def get_execution_outputs(execution: GraphExecution | None) -> dict[str, Any] | None:
    """Extract outputs from an execution, or return None."""
    if execution is None:
        return None
    return execution.outputs
