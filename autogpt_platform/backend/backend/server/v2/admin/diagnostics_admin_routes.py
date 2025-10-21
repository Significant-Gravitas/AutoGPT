import logging
from datetime import datetime, timezone

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, HTTPException, Security
from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraphExecution, AgentGraph

from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor.utils import create_execution_queue_config, GRAPH_EXECUTION_QUEUE_NAME
from backend.server.v2.admin.model import (
    ExecutionDiagnosticsResponse,
    AgentDiagnosticsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["diagnostics", "admin"],
    dependencies=[Security(requires_admin_user)],
)


async def get_running_executions_count() -> int:
    """Get the count of currently running executions from the database."""
    try:
        count = await AgentGraphExecution.prisma().count(
            where={"executionStatus": AgentExecutionStatus.RUNNING}
        )
        return count
    except Exception as e:
        logger.error(f"Error getting running executions count: {e}")
        raise


async def get_queued_executions_db_count() -> int:
    """Get the count of queued executions from the database."""
    try:
        count = await AgentGraphExecution.prisma().count(
            where={"executionStatus": AgentExecutionStatus.QUEUED}
        )
        return count
    except Exception as e:
        logger.error(f"Error getting queued executions count from DB: {e}")
        raise


def get_rabbitmq_queue_depth() -> int:
    """Get the number of messages in the RabbitMQ execution queue."""
    try:
        # Create a temporary connection to query the queue
        config = create_execution_queue_config()
        rabbitmq = SyncRabbitMQ(config)
        rabbitmq.connect()

        # Use passive queue_declare to get queue info without modifying it
        method_frame = rabbitmq._channel.queue_declare(
            queue=GRAPH_EXECUTION_QUEUE_NAME,
            passive=True
        )

        message_count = method_frame.method.message_count

        # Clean up connection
        rabbitmq.disconnect()

        return message_count
    except Exception as e:
        logger.error(f"Error getting RabbitMQ queue depth: {e}")
        # Return -1 to indicate an error state rather than failing the entire request
        return -1


async def get_agents_with_active_executions_count() -> int:
    """Get the count of unique agents that have running or queued executions."""
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
        return len(executions)
    except Exception as e:
        logger.error(f"Error getting agents with active executions count: {e}")
        raise


@router.get(
    "/diagnostics/executions",
    response_model=ExecutionDiagnosticsResponse,
    summary="Get Execution Diagnostics",
)
async def get_execution_diagnostics():
    """
    Get diagnostic information about execution status.

    Returns:
        - running_executions: Number of executions currently running
        - queued_executions_db: Number of executions queued in the database
        - queued_executions_rabbitmq: Number of messages in the RabbitMQ queue (-1 if error)
        - timestamp: Current timestamp
    """
    try:
        logger.info("Getting execution diagnostics")

        running_count = await get_running_executions_count()
        queued_db_count = await get_queued_executions_db_count()
        rabbitmq_count = get_rabbitmq_queue_depth()

        response = ExecutionDiagnosticsResponse(
            running_executions=running_count,
            queued_executions_db=queued_db_count,
            queued_executions_rabbitmq=rabbitmq_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            f"Execution diagnostics: running={running_count}, "
            f"queued_db={queued_db_count}, queued_rabbitmq={rabbitmq_count}"
        )

        return response
    except Exception as e:
        logger.exception(f"Error getting execution diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/diagnostics/agents",
    response_model=AgentDiagnosticsResponse,
    summary="Get Agent Diagnostics",
)
async def get_agent_diagnostics():
    """
    Get diagnostic information about agents.

    Returns:
        - agents_with_active_executions: Number of unique agents with running/queued executions
        - timestamp: Current timestamp
    """
    try:
        logger.info("Getting agent diagnostics")

        active_executions_count = await get_agents_with_active_executions_count()

        response = AgentDiagnosticsResponse(
            agents_with_active_executions=active_executions_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            f"Agent diagnostics: with_active_executions={active_executions_count}"
        )

        return response
    except Exception as e:
        logger.exception(f"Error getting agent diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
