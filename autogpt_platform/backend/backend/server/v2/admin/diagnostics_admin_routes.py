import logging
from typing import List

from autogpt_libs.auth import requires_admin_user
from autogpt_libs.auth.models import User as AuthUser
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel

from backend.data.diagnostics import (
    RunningExecutionDetail,
    get_agent_diagnostics,
    get_execution_diagnostics,
    get_running_executions_details,
    stop_execution,
    stop_executions_bulk,
)
from backend.server.v2.admin.model import (
    AgentDiagnosticsResponse,
    ExecutionDiagnosticsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["diagnostics", "admin"],
    dependencies=[Security(requires_admin_user)],
)


class RunningExecutionsListResponse(BaseModel):
    """Response model for list of running executions"""

    executions: List[RunningExecutionDetail]
    total: int


class StopExecutionRequest(BaseModel):
    """Request model for stopping a single execution"""

    execution_id: str


class StopExecutionsRequest(BaseModel):
    """Request model for stopping multiple executions"""

    execution_ids: List[str]


class StopExecutionResponse(BaseModel):
    """Response model for stop execution operations"""

    success: bool
    stopped_count: int = 0
    message: str


@router.get(
    "/diagnostics/executions",
    response_model=ExecutionDiagnosticsResponse,
    summary="Get Execution Diagnostics",
)
async def get_execution_diagnostics_endpoint():
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

        diagnostics = await get_execution_diagnostics()

        response = ExecutionDiagnosticsResponse(
            running_executions=diagnostics.running_count,
            queued_executions_db=diagnostics.queued_db_count,
            queued_executions_rabbitmq=diagnostics.rabbitmq_queue_depth,
            timestamp=diagnostics.timestamp,
        )

        logger.info(
            f"Execution diagnostics: running={diagnostics.running_count}, "
            f"queued_db={diagnostics.queued_db_count}, queued_rabbitmq={diagnostics.rabbitmq_queue_depth}"
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
async def get_agent_diagnostics_endpoint():
    """
    Get diagnostic information about agents.

    Returns:
        - agents_with_active_executions: Number of unique agents with running/queued executions
        - timestamp: Current timestamp
    """
    try:
        logger.info("Getting agent diagnostics")

        diagnostics = await get_agent_diagnostics()

        response = AgentDiagnosticsResponse(
            agents_with_active_executions=diagnostics.agents_with_active_executions,
            timestamp=diagnostics.timestamp,
        )

        logger.info(
            f"Agent diagnostics: with_active_executions={diagnostics.agents_with_active_executions}"
        )

        return response
    except Exception as e:
        logger.exception(f"Error getting agent diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/diagnostics/executions/running",
    response_model=RunningExecutionsListResponse,
    summary="List Running Executions",
)
async def list_running_executions(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get detailed list of running and queued executions.

    Args:
        limit: Maximum number of executions to return (default 100)
        offset: Number of executions to skip (default 0)

    Returns:
        List of running executions with details
    """
    try:
        logger.info(f"Listing running executions (limit={limit}, offset={offset})")

        executions = await get_running_executions_details(limit=limit, offset=offset)

        # Get total count for pagination
        from backend.data.diagnostics import get_execution_diagnostics as get_diag

        diagnostics = await get_diag()
        total = diagnostics.running_count + diagnostics.queued_db_count

        return RunningExecutionsListResponse(executions=executions, total=total)
    except Exception as e:
        logger.exception(f"Error listing running executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/diagnostics/executions/stop",
    response_model=StopExecutionResponse,
    summary="Stop Single Execution",
)
async def stop_single_execution(
    request: StopExecutionRequest,
    user: AuthUser = Security(requires_admin_user),
):
    """
    Stop a single execution (admin only).

    Args:
        request: Contains execution_id to stop

    Returns:
        Success status and message
    """
    try:
        logger.info(f"Admin {user.id} stopping execution {request.execution_id}")

        success = await stop_execution(request.execution_id, user.id)

        return StopExecutionResponse(
            success=success,
            stopped_count=1 if success else 0,
            message=(
                "Execution stopped successfully"
                if success
                else "Failed to stop execution"
            ),
        )
    except Exception as e:
        logger.exception(f"Error stopping execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/diagnostics/executions/stop-bulk",
    response_model=StopExecutionResponse,
    summary="Stop Multiple Executions",
)
async def stop_multiple_executions(
    request: StopExecutionsRequest,
    user: AuthUser = Security(requires_admin_user),
):
    """
    Stop multiple executions (admin only).

    Args:
        request: Contains list of execution_ids to stop

    Returns:
        Number of executions stopped and success message
    """
    try:
        logger.info(f"Admin {user.id} stopping {len(request.execution_ids)} executions")

        stopped_count = await stop_executions_bulk(request.execution_ids, user.id)

        return StopExecutionResponse(
            success=stopped_count > 0,
            stopped_count=stopped_count,
            message=f"Stopped {stopped_count} of {len(request.execution_ids)} executions",
        )
    except Exception as e:
        logger.exception(f"Error stopping multiple executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
