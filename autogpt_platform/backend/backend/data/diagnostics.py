"""
Diagnostics module for monitoring and troubleshooting execution status.
"""

import logging
from datetime import datetime
from typing import Optional

from prisma.models import AgentGraphExecution
from pydantic import BaseModel

from backend.data.execution import ExecutionStatus

logger = logging.getLogger(__name__)


class RunningExecutionDetails(BaseModel):
    """Details about a running execution for diagnostics."""

    execution_id: str
    graph_id: str
    graph_name: str
    graph_version: int
    user_id: str
    user_email: Optional[str]
    status: str
    started_at: Optional[datetime]
    queue_status: Optional[str] = None


class ExecutionDiagnostics(BaseModel):
    """Overall execution diagnostics information."""

    total_running: int
    total_queued: int
    total_incomplete: int


async def get_running_executions_details(
    limit: int = 10,
    offset: int = 0,
) -> list[RunningExecutionDetails]:
    """
    Get detailed information about currently running executions.

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip

    Returns:
        List of running execution details

    Raises:
        Exception: If there's an error retrieving execution details
    """
    try:
        # Query for running and queued executions
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "isDeleted": False,
                "OR": [
                    {"executionStatus": ExecutionStatus.RUNNING},
                    {"executionStatus": ExecutionStatus.QUEUED},
                ],
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            order={"createdAt": "desc"},
            skip=offset,
            take=limit,
        )

        result = []
        for exec in executions:
            # Convert string executionStatus to enum if needed, then to string for response
            # The database field executionStatus is a string, not an enum
            status_value = exec.executionStatus
            if isinstance(status_value, str):
                # It's already a string, use it directly
                status_str = status_value
            else:
                # It's an enum, get the value
                status_str = status_value.value

            result.append(
                RunningExecutionDetails(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=exec.AgentGraph.name if exec.AgentGraph else "Unknown",
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=status_str,
                    started_at=exec.startedAt,
                    queue_status=(
                        exec.queueStatus if hasattr(exec, "queueStatus") else None
                    ),
                )
            )

        return result

    except Exception as e:
        logger.error(f"Error getting running execution details: {e}")
        raise


async def get_execution_diagnostics() -> ExecutionDiagnostics:
    """
    Get overall execution diagnostics information.

    Returns:
        ExecutionDiagnostics with counts of executions by status
    """
    try:
        running_count = await AgentGraphExecution.prisma().count(
            where={
                "isDeleted": False,
                "executionStatus": ExecutionStatus.RUNNING,
            }
        )

        queued_count = await AgentGraphExecution.prisma().count(
            where={
                "isDeleted": False,
                "executionStatus": ExecutionStatus.QUEUED,
            }
        )

        incomplete_count = await AgentGraphExecution.prisma().count(
            where={
                "isDeleted": False,
                "executionStatus": ExecutionStatus.INCOMPLETE,
            }
        )

        return ExecutionDiagnostics(
            total_running=running_count,
            total_queued=queued_count,
            total_incomplete=incomplete_count,
        )

    except Exception as e:
        logger.error(f"Error getting execution diagnostics: {e}")
        raise