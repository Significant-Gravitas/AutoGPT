"""
Diagnostics data layer for admin operations.
Provides functions to query and manage system diagnostics including executions and agents.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraphExecution
from pydantic import BaseModel

from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor.utils import (
    GRAPH_EXECUTION_QUEUE_NAME,
    create_execution_queue_config,
)

logger = logging.getLogger(__name__)


class RunningExecutionDetail(BaseModel):
    """Details about a running execution for admin view"""

    execution_id: str
    graph_id: str
    graph_name: str
    graph_version: int
    user_id: str
    user_email: Optional[str]
    status: str
    started_at: Optional[datetime]
    queue_status: Optional[str]


class ExecutionDiagnosticsSummary(BaseModel):
    """Summary of execution diagnostics"""

    running_count: int
    queued_db_count: int
    rabbitmq_queue_depth: int
    timestamp: str


class AgentDiagnosticsSummary(BaseModel):
    """Summary of agent diagnostics"""

    agents_with_active_executions: int
    timestamp: str


async def get_execution_diagnostics() -> ExecutionDiagnosticsSummary:
    """
    Get comprehensive execution diagnostics including database and queue metrics.

    Returns:
        ExecutionDiagnosticsSummary with current execution state
    """
    try:
        # Get running executions count
        running_count = await AgentGraphExecution.prisma().count(
            where={"executionStatus": AgentExecutionStatus.RUNNING}
        )

        # Get queued executions from database
        queued_db_count = await AgentGraphExecution.prisma().count(
            where={"executionStatus": AgentExecutionStatus.QUEUED}
        )

        # Get RabbitMQ queue depth
        rabbitmq_queue_depth = get_rabbitmq_queue_depth()

        return ExecutionDiagnosticsSummary(
            running_count=running_count,
            queued_db_count=queued_db_count,
            rabbitmq_queue_depth=rabbitmq_queue_depth,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting execution diagnostics: {e}")
        raise


async def get_agent_diagnostics() -> AgentDiagnosticsSummary:
    """
    Get comprehensive agent diagnostics.

    Returns:
        AgentDiagnosticsSummary with agent metrics
    """
    try:
        # Get distinct agent graph IDs with active executions
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": {
                    "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]
                }
            },
            distinct=["agentGraphId"],
        )

        return AgentDiagnosticsSummary(
            agents_with_active_executions=len(executions),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting agent diagnostics: {e}")
        raise


def get_rabbitmq_queue_depth() -> int:
    """
    Get the number of messages in the RabbitMQ execution queue.

    Returns:
        Number of messages in queue, or -1 if error
    """
    try:
        # Create a temporary connection to query the queue
        config = create_execution_queue_config()
        rabbitmq = SyncRabbitMQ(config)
        rabbitmq.connect()

        # Use passive queue_declare to get queue info without modifying it
        method_frame = rabbitmq._channel.queue_declare(
            queue=GRAPH_EXECUTION_QUEUE_NAME, passive=True
        )

        message_count = method_frame.method.message_count

        # Clean up connection
        rabbitmq.disconnect()

        return message_count
    except Exception as e:
        logger.error(f"Error getting RabbitMQ queue depth: {e}")
        # Return -1 to indicate an error state rather than failing the entire request
        return -1


async def get_running_executions_details(
    limit: int = 100, offset: int = 0
) -> List[RunningExecutionDetail]:
    """
    Get detailed information about running executions.

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip

    Returns:
        List of RunningExecutionDetail objects
    """
    try:
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": {
                    "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]
                }
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            take=limit,
            skip=offset,
            order={"createdAt": "desc"},
        )

        results = []
        for exec in executions:
            results.append(
                RunningExecutionDetail(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=exec.AgentGraph.name if exec.AgentGraph else "Unknown",
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=exec.executionStatus.value,
                    started_at=exec.startedAt,
                    queue_status=(
                        exec.queueStatus if hasattr(exec, "queueStatus") else None
                    ),
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting running execution details: {e}")
        raise


async def stop_execution(execution_id: str, admin_user_id: str) -> bool:
    """
    Stop a single execution by setting its status to FAILED.
    Admin-only operation.

    Args:
        execution_id: ID of the execution to stop
        admin_user_id: ID of the admin user performing the operation

    Returns:
        True if execution was stopped, False otherwise
    """
    try:
        logger.info(f"Admin user {admin_user_id} stopping execution {execution_id}")

        # Update the execution status to FAILED
        result = await AgentGraphExecution.prisma().update(
            where={"id": execution_id},
            data={
                "executionStatus": AgentExecutionStatus.FAILED,
                "error": "Execution stopped by admin",
                "updatedAt": datetime.now(timezone.utc),
            },
        )

        return result is not None
    except Exception as e:
        logger.error(f"Error stopping execution {execution_id}: {e}")
        return False


async def stop_executions_bulk(execution_ids: List[str], admin_user_id: str) -> int:
    """
    Stop multiple executions by setting their status to FAILED.
    Admin-only operation.

    Args:
        execution_ids: List of execution IDs to stop
        admin_user_id: ID of the admin user performing the operation

    Returns:
        Number of executions successfully stopped
    """
    try:
        logger.info(
            f"Admin user {admin_user_id} stopping {len(execution_ids)} executions"
        )

        # Update all executions to FAILED status
        result = await AgentGraphExecution.prisma().update_many(
            where={
                "id": {"in": execution_ids},
                "executionStatus": {
                    "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]
                },
            },
            data={
                "executionStatus": AgentExecutionStatus.FAILED,
                "error": "Execution stopped by admin",
                "updatedAt": datetime.now(timezone.utc),
            },
        )

        return result
    except Exception as e:
        logger.error(f"Error stopping executions in bulk: {e}")
        return 0
