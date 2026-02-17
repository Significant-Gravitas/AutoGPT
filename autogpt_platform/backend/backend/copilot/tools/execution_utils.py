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


async def wait_for_execution(
    user_id: str,
    graph_id: str,
    execution_id: str,
    timeout_seconds: int,
) -> GraphExecution | None:
    """
    Wait for an execution to reach a terminal status using Redis pubsub.

    Uses asyncio.wait_for to ensure timeout is respected even when no events
    are received.

    Args:
        user_id: User ID
        graph_id: Graph ID
        execution_id: Execution ID to wait for
        timeout_seconds: Max seconds to wait

    Returns:
        The execution with current status, or None if not found
    """
    exec_db = execution_db()

    # First check current status - maybe it's already done
    execution = await exec_db.get_graph_execution(
        user_id=user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )
    if not execution:
        return None

    # If already in terminal state, return immediately
    if execution.status in TERMINAL_STATUSES:
        logger.debug(
            f"Execution {execution_id} already in terminal state: {execution.status}"
        )
        return execution

    logger.info(
        f"Waiting up to {timeout_seconds}s for execution {execution_id} "
        f"(current status: {execution.status})"
    )

    # Subscribe to execution updates via Redis pubsub
    event_bus = AsyncRedisExecutionEventBus()
    channel_key = f"{user_id}/{graph_id}/{execution_id}"

    try:
        # Use wait_for to enforce timeout on the entire listen operation
        result = await asyncio.wait_for(
            _listen_for_terminal_status(
                event_bus, channel_key, user_id, execution_id, exec_db
            ),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        logger.info(f"Timeout waiting for execution {execution_id}")
    except Exception as e:
        logger.error(f"Error waiting for execution: {e}", exc_info=True)

    # Return current state on timeout/error
    return await exec_db.get_graph_execution(
        user_id=user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )


async def _listen_for_terminal_status(
    event_bus: AsyncRedisExecutionEventBus,
    channel_key: str,
    user_id: str,
    execution_id: str,
    exec_db: Any,
) -> GraphExecution | None:
    """
    Listen for execution events until a terminal status is reached.

    This is a helper that gets wrapped in asyncio.wait_for for timeout handling.
    """
    async for event in event_bus.listen_events(channel_key):
        # Only process GraphExecutionEvents (not NodeExecutionEvents)
        if isinstance(event, GraphExecutionEvent):
            logger.debug(f"Received execution update: {event.status}")
            if event.status in TERMINAL_STATUSES:
                # Fetch full execution with outputs
                return await exec_db.get_graph_execution(
                    user_id=user_id,
                    execution_id=execution_id,
                    include_node_executions=False,
                )

    # Should not reach here normally (generator should yield indefinitely)
    return None


def get_execution_outputs(execution: GraphExecution | None) -> dict[str, Any] | None:
    """Extract outputs from an execution, or return None."""
    if execution is None:
        return None
    return execution.outputs
