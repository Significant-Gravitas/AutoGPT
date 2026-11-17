"""Shared utilities for execution waiting and status handling."""

import asyncio
import logging
from contextlib import aclosing
from typing import Any, Sequence

from typing_extensions import TypedDict

from backend.blocks import get_block
from backend.data.db_accessors import execution_db
from backend.data.execution import (
    AsyncRedisExecutionEventBus,
    ExecutionStatus,
    GraphExecution,
    GraphExecutionEvent,
    NodeExecutionResult,
    exec_channel,
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
    channel_key = exec_channel(user_id, graph_id, execution_id)

    try:
        async with asyncio.timeout(timeout_seconds):
            return await _subscribe_and_wait(
                event_bus, channel_key, user_id, execution_id, exec_db
            )
    except TimeoutError:
        logger.info(f"Timeout waiting for execution {execution_id}")
    except Exception as e:
        logger.error(f"Error waiting for execution: {e}", exc_info=True)

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
    """Wait on execution events for a terminal or paused status.

    ``listen_with_resync`` yields ``None`` once each (re)subscription is
    confirmed, and once when the stream is lost; the database is checked
    then, so a status reached before the subscription or while the stream
    was down is still seen.
    """
    async with aclosing(event_bus.listen_with_resync(channel_key)) as events:
        async for event in events:
            stop_event = (
                isinstance(event, GraphExecutionEvent)
                and event.status in STOP_WAITING_STATUSES
            )
            if event is not None and not stop_event:
                continue
            execution = await exec_db.get_graph_execution(
                user_id=user_id,
                execution_id=execution_id,
                include_node_executions=False,
            )
            if stop_event or (execution and execution.status in STOP_WAITING_STATUSES):
                return execution
    return None


def get_execution_outputs(execution: GraphExecution | None) -> dict[str, Any] | None:
    """Extract outputs from an execution, or return None."""
    if execution is None:
        return None
    return execution.outputs


class NodeFailureSummary(TypedDict):
    node_id: str
    block_name: str
    error: str | None


# Per-node error detail cap: enough to identify the failure, small enough
# that a many-failure run can't flood the tool response.
_NODE_ERROR_MAX_CHARS = 500


def summarize_node_failures(
    node_executions: Sequence[NodeExecutionResult],
) -> list[NodeFailureSummary]:
    """Compact per-node failure summary for a graph execution.

    A graph run can finish with status COMPLETED while individual nodes
    FAILED — node errors don't fail the run. Surfacing them explicitly
    prevents the assistant from reporting success based on the top-level
    status alone.
    """
    failures: list[NodeFailureSummary] = []
    for ne in node_executions:
        if ne.status != ExecutionStatus.FAILED:
            continue
        block = get_block(ne.block_id)
        error_values = ne.output_data.get("error") if ne.output_data else None
        if isinstance(error_values, list):
            error_str = str(error_values[0]) if error_values else None
        elif error_values:
            error_str = str(error_values)
        else:
            error_str = None
        failures.append(
            {
                "node_id": ne.node_id,
                "block_name": block.name if block else ne.block_id,
                "error": error_str[:_NODE_ERROR_MAX_CHARS] if error_str else None,
            }
        )
    return failures


def build_run_health_warning(
    outputs: dict[str, Any] | None,
    node_failures: list[NodeFailureSummary],
    failed_node_count: int = 0,
) -> str | None:
    """Warning to include in a COMPLETED-run message, or None if healthy.

    `failed_node_count` is the run's own tally (`stats.node_error_count`). It
    still flags a failed node when the per-node trace that `node_failures` is
    built from wasn't loaded.
    """
    if node_failures:
        details = "; ".join(
            f"{f['block_name']}: {f['error'] or 'no error detail'}"
            for f in node_failures[:5]
        )
        return (
            f"WARNING: {len(node_failures)} node(s) FAILED despite the COMPLETED "
            f"status ({details}). Treat this run as unsuccessful — inspect "
            "node_executions, fix the workflow, and re-verify before reporting "
            "success."
        )
    if failed_node_count > 0:
        return (
            f"WARNING: {failed_node_count} node(s) FAILED despite the COMPLETED "
            "status. Treat this run as unsuccessful — inspect node_executions, "
            "fix the workflow, and re-verify before reporting success."
        )
    if not outputs:
        return (
            "WARNING: the run COMPLETED but produced no outputs. If this "
            "workflow declares AgentOutput blocks, that usually means a broken "
            "link or a node that silently produced nothing — inspect "
            "node_executions before reporting success. (Side-effect-only "
            "workflows with no declared outputs legitimately finish this way.)"
        )
    return None
