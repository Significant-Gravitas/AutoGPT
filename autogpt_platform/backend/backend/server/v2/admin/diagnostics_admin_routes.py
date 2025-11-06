import logging
from typing import List

from autogpt_libs.auth import requires_admin_user
from autogpt_libs.auth.models import User as AuthUser
from fastapi import APIRouter, Security
from pydantic import BaseModel

from backend.data.diagnostics import (
    FailedExecutionDetail,
    OrphanedScheduleDetail,
    RunningExecutionDetail,
    ScheduleDetail,
    ScheduleHealthMetrics,
    cleanup_all_stuck_queued_executions,
    cleanup_orphaned_executions_bulk,
    cleanup_orphaned_schedules_bulk,
    get_agent_diagnostics,
    get_all_orphaned_execution_ids,
    get_all_schedules_details,
    get_all_stuck_queued_execution_ids,
    get_execution_diagnostics,
    get_failed_executions_count,
    get_failed_executions_details,
    get_long_running_executions_details,
    get_orphaned_executions_details,
    get_orphaned_schedules_details,
    get_running_executions_details,
    get_schedule_health_metrics,
    get_stuck_queued_executions_details,
    requeue_execution,
    requeue_executions_bulk,
    stop_all_long_running_executions,
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


class FailedExecutionsListResponse(BaseModel):
    """Response model for list of failed executions"""

    executions: List[FailedExecutionDetail]
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


class RequeueExecutionResponse(BaseModel):
    """Response model for requeue execution operations"""

    success: bool
    requeued_count: int = 0
    message: str


@router.get(
    "/diagnostics/executions",
    response_model=ExecutionDiagnosticsResponse,
    summary="Get Execution Diagnostics",
)
async def get_execution_diagnostics_endpoint():
    """
    Get comprehensive diagnostic information about execution status.

    Returns all execution metrics including:
    - Current state (running, queued)
    - Orphaned executions (>24h old, likely not in executor)
    - Failure metrics (1h, 24h, rate)
    - Long-running detection (stuck >1h, >24h)
    - Stuck queued detection
    - Throughput metrics (completions/hour)
    - RabbitMQ queue depths
    """
    logger.info("Getting execution diagnostics")

    diagnostics = await get_execution_diagnostics()

    response = ExecutionDiagnosticsResponse(
        running_executions=diagnostics.running_count,
        queued_executions_db=diagnostics.queued_db_count,
        queued_executions_rabbitmq=diagnostics.rabbitmq_queue_depth,
        cancel_queue_depth=diagnostics.cancel_queue_depth,
        orphaned_running=diagnostics.orphaned_running,
        orphaned_queued=diagnostics.orphaned_queued,
        failed_count_1h=diagnostics.failed_count_1h,
        failed_count_24h=diagnostics.failed_count_24h,
        failure_rate_24h=diagnostics.failure_rate_24h,
        stuck_running_24h=diagnostics.stuck_running_24h,
        stuck_running_1h=diagnostics.stuck_running_1h,
        oldest_running_hours=diagnostics.oldest_running_hours,
        stuck_queued_1h=diagnostics.stuck_queued_1h,
        queued_never_started=diagnostics.queued_never_started,
        completed_1h=diagnostics.completed_1h,
        completed_24h=diagnostics.completed_24h,
        throughput_per_hour=diagnostics.throughput_per_hour,
        timestamp=diagnostics.timestamp,
    )

    logger.info(
        f"Execution diagnostics: running={diagnostics.running_count}, "
        f"queued_db={diagnostics.queued_db_count}, "
        f"orphaned={diagnostics.orphaned_running + diagnostics.orphaned_queued}, "
        f"failed_24h={diagnostics.failed_count_24h}"
    )

    return response


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
    Get detailed list of running and queued executions (recent, likely active).

    Args:
        limit: Maximum number of executions to return (default 100)
        offset: Number of executions to skip (default 0)

    Returns:
        List of running executions with details
    """
    logger.info(f"Listing running executions (limit={limit}, offset={offset})")

    executions = await get_running_executions_details(limit=limit, offset=offset)

    # Get total count for pagination
    diagnostics = await get_execution_diagnostics()
    total = diagnostics.running_count + diagnostics.queued_db_count

    return RunningExecutionsListResponse(executions=executions, total=total)


@router.get(
    "/diagnostics/executions/orphaned",
    response_model=RunningExecutionsListResponse,
    summary="List Orphaned Executions",
)
async def list_orphaned_executions(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get detailed list of orphaned executions (>24h old, likely not in executor).

    Args:
        limit: Maximum number of executions to return (default 100)
        offset: Number of executions to skip (default 0)

    Returns:
        List of orphaned executions with details
    """
    logger.info(f"Listing orphaned executions (limit={limit}, offset={offset})")

    executions = await get_orphaned_executions_details(limit=limit, offset=offset)

    # Get total count for pagination
    diagnostics = await get_execution_diagnostics()
    total = diagnostics.orphaned_running + diagnostics.orphaned_queued

    return RunningExecutionsListResponse(executions=executions, total=total)


@router.get(
    "/diagnostics/executions/failed",
    response_model=FailedExecutionsListResponse,
    summary="List Failed Executions",
)
async def list_failed_executions(
    limit: int = 100,
    offset: int = 0,
    hours: int = 24,
):
    """
    Get detailed list of failed executions.

    Args:
        limit: Maximum number of executions to return (default 100)
        offset: Number of executions to skip (default 0)
        hours: Number of hours to look back (default 24)

    Returns:
        List of failed executions with error details
    """
    logger.info(
        f"Listing failed executions (limit={limit}, offset={offset}, hours={hours})"
    )

    executions = await get_failed_executions_details(
        limit=limit, offset=offset, hours=hours
    )

    # Get total count for pagination
    # Always count actual total for given hours parameter
    total = await get_failed_executions_count(hours=hours)

    return FailedExecutionsListResponse(executions=executions, total=total)


@router.get(
    "/diagnostics/executions/long-running",
    response_model=RunningExecutionsListResponse,
    summary="List Long-Running Executions",
)
async def list_long_running_executions(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get detailed list of long-running executions (RUNNING status >24h).

    Args:
        limit: Maximum number of executions to return (default 100)
        offset: Number of executions to skip (default 0)

    Returns:
        List of long-running executions with details
    """
    logger.info(f"Listing long-running executions (limit={limit}, offset={offset})")

    executions = await get_long_running_executions_details(limit=limit, offset=offset)

    # Get total count for pagination
    diagnostics = await get_execution_diagnostics()
    total = diagnostics.stuck_running_24h

    return RunningExecutionsListResponse(executions=executions, total=total)


@router.get(
    "/diagnostics/executions/stuck-queued",
    response_model=RunningExecutionsListResponse,
    summary="List Stuck Queued Executions",
)
async def list_stuck_queued_executions(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get detailed list of stuck queued executions (QUEUED >1h, never started).

    Args:
        limit: Maximum number of executions to return (default 100)
        offset: Number of executions to skip (default 0)

    Returns:
        List of stuck queued executions with details
    """
    logger.info(f"Listing stuck queued executions (limit={limit}, offset={offset})")

    executions = await get_stuck_queued_executions_details(limit=limit, offset=offset)

    # Get total count for pagination
    diagnostics = await get_execution_diagnostics()
    total = diagnostics.stuck_queued_1h

    return RunningExecutionsListResponse(executions=executions, total=total)


@router.post(
    "/diagnostics/executions/requeue",
    response_model=RequeueExecutionResponse,
    summary="Requeue Stuck Execution",
)
async def requeue_single_execution(
    request: StopExecutionRequest,  # Reuse same request model (has execution_id)
    user: AuthUser = Security(requires_admin_user),
):
    """
    Requeue a stuck QUEUED execution (admin only).
    Publishes execution to RabbitMQ queue.

    ⚠️ WARNING: Only use for stuck executions. This will re-execute and may cost credits.

    Args:
        request: Contains execution_id to requeue

    Returns:
        Success status and message
    """
    logger.info(f"Admin {user.user_id} requeueing execution {request.execution_id}")

    success = await requeue_execution(request.execution_id, user.user_id)

    return RequeueExecutionResponse(
        success=success,
        requeued_count=1 if success else 0,
        message=(
            "Execution requeued successfully"
            if success
            else "Failed to requeue execution"
        ),
    )


@router.post(
    "/diagnostics/executions/requeue-bulk",
    response_model=RequeueExecutionResponse,
    summary="Requeue Multiple Stuck Executions",
)
async def requeue_multiple_executions(
    request: StopExecutionsRequest,  # Reuse same request model (has execution_ids)
    user: AuthUser = Security(requires_admin_user),
):
    """
    Requeue multiple stuck QUEUED executions (admin only).
    Publishes executions to RabbitMQ queue.

    ⚠️ WARNING: Only use for stuck executions. This will re-execute and may cost credits.

    Args:
        request: Contains list of execution_ids to requeue

    Returns:
        Number of executions requeued and success message
    """
    logger.info(
        f"Admin {user.user_id} requeueing {len(request.execution_ids)} executions"
    )

    requeued_count = await requeue_executions_bulk(request.execution_ids, user.user_id)

    return RequeueExecutionResponse(
        success=requeued_count > 0,
        requeued_count=requeued_count,
        message=f"Requeued {requeued_count} of {len(request.execution_ids)} executions",
    )


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
    logger.info(f"Admin {user.user_id} stopping execution {request.execution_id}")

    success = await stop_execution(request.execution_id, user.user_id)

    return StopExecutionResponse(
        success=success,
        stopped_count=1 if success else 0,
        message=(
            "Execution stopped successfully" if success else "Failed to stop execution"
        ),
    )


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
    Stop multiple active executions (admin only).
    Sends cancel signals to executor for recent executions.

    Args:
        request: Contains list of execution_ids to stop

    Returns:
        Number of executions stopped and success message
    """
    logger.info(
        f"Admin {user.user_id} stopping {len(request.execution_ids)} executions"
    )

    stopped_count = await stop_executions_bulk(request.execution_ids, user.user_id)

    return StopExecutionResponse(
        success=stopped_count > 0,
        stopped_count=stopped_count,
        message=f"Stopped {stopped_count} of {len(request.execution_ids)} executions",
    )


@router.post(
    "/diagnostics/executions/cleanup-orphaned",
    response_model=StopExecutionResponse,
    summary="Cleanup Orphaned Executions",
)
async def cleanup_orphaned_executions(
    request: StopExecutionsRequest,
    user: AuthUser = Security(requires_admin_user),
):
    """
    Cleanup orphaned executions by directly updating DB status (admin only).
    For executions in DB but not actually running in executor (old/stale records).

    Args:
        request: Contains list of execution_ids to cleanup

    Returns:
        Number of executions cleaned up and success message
    """
    logger.info(
        f"Admin {user.user_id} cleaning up {len(request.execution_ids)} orphaned executions"
    )

    cleaned_count = await cleanup_orphaned_executions_bulk(
        request.execution_ids, user.user_id
    )

    return StopExecutionResponse(
        success=cleaned_count > 0,
        stopped_count=cleaned_count,
        message=f"Cleaned up {cleaned_count} of {len(request.execution_ids)} orphaned executions",
    )


# ============================================================================
# SCHEDULE DIAGNOSTICS ENDPOINTS
# ============================================================================


class SchedulesListResponse(BaseModel):
    """Response model for list of schedules"""

    schedules: List[ScheduleDetail]
    total: int


class OrphanedSchedulesListResponse(BaseModel):
    """Response model for list of orphaned schedules"""

    schedules: List[OrphanedScheduleDetail]
    total: int


class ScheduleCleanupResponse(BaseModel):
    """Response model for schedule cleanup operations"""

    success: bool
    deleted_count: int = 0
    message: str


@router.get(
    "/diagnostics/schedules",
    response_model=ScheduleHealthMetrics,
    summary="Get Schedule Diagnostics",
)
async def get_schedule_diagnostics_endpoint():
    """
    Get comprehensive diagnostic information about schedule health.

    Returns schedule metrics including:
    - Total schedules (user vs system)
    - Orphaned schedules by category
    - Upcoming executions
    """
    logger.info("Getting schedule diagnostics")

    diagnostics = await get_schedule_health_metrics()

    logger.info(
        f"Schedule diagnostics: total={diagnostics.total_schedules}, "
        f"user={diagnostics.user_schedules}, "
        f"orphaned={diagnostics.total_orphaned}"
    )

    return diagnostics


@router.get(
    "/diagnostics/schedules/all",
    response_model=SchedulesListResponse,
    summary="List All User Schedules",
)
async def list_all_schedules(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get detailed list of all user schedules (excludes system monitoring jobs).

    Args:
        limit: Maximum number of schedules to return (default 100)
        offset: Number of schedules to skip (default 0)

    Returns:
        List of schedules with details
    """
    logger.info(f"Listing all schedules (limit={limit}, offset={offset})")

    schedules = await get_all_schedules_details(limit=limit, offset=offset)

    # Get total count
    diagnostics = await get_schedule_health_metrics()
    total = diagnostics.user_schedules

    return SchedulesListResponse(schedules=schedules, total=total)


@router.get(
    "/diagnostics/schedules/orphaned",
    response_model=OrphanedSchedulesListResponse,
    summary="List Orphaned Schedules",
)
async def list_orphaned_schedules():
    """
    Get detailed list of orphaned schedules with orphan reasons.

    Returns:
        List of orphaned schedules categorized by orphan type
    """
    logger.info("Listing orphaned schedules")

    schedules = await get_orphaned_schedules_details()

    return OrphanedSchedulesListResponse(schedules=schedules, total=len(schedules))


@router.post(
    "/diagnostics/schedules/cleanup-orphaned",
    response_model=ScheduleCleanupResponse,
    summary="Cleanup Orphaned Schedules",
)
async def cleanup_orphaned_schedules(
    request: StopExecutionsRequest,  # Reuse for schedule_ids list
    user: AuthUser = Security(requires_admin_user),
):
    """
    Cleanup orphaned schedules by deleting from scheduler (admin only).

    Args:
        request: Contains list of schedule_ids to delete

    Returns:
        Number of schedules deleted and success message
    """
    logger.info(
        f"Admin {user.user_id} cleaning up {len(request.execution_ids)} orphaned schedules"
    )

    deleted_count = await cleanup_orphaned_schedules_bulk(
        request.execution_ids, user.user_id
    )

    return ScheduleCleanupResponse(
        success=deleted_count > 0,
        deleted_count=deleted_count,
        message=f"Deleted {deleted_count} of {len(request.execution_ids)} orphaned schedules",
    )


@router.post(
    "/diagnostics/executions/stop-all-long-running",
    response_model=StopExecutionResponse,
    summary="Stop ALL Long-Running Executions",
)
async def stop_all_long_running_executions_endpoint(
    user: AuthUser = Security(requires_admin_user),
):
    """
    Stop ALL long-running executions (RUNNING >24h) by sending cancel signals (admin only).
    Operates on entire dataset, not limited to pagination.

    Returns:
        Number of executions stopped and success message
    """
    logger.info(f"Admin {user.user_id} stopping ALL long-running executions")

    stopped_count = await stop_all_long_running_executions(user.user_id)

    return StopExecutionResponse(
        success=stopped_count > 0,
        stopped_count=stopped_count,
        message=f"Stopped {stopped_count} long-running executions",
    )


@router.post(
    "/diagnostics/executions/cleanup-all-orphaned",
    response_model=StopExecutionResponse,
    summary="Cleanup ALL Orphaned Executions",
)
async def cleanup_all_orphaned_executions(
    user: AuthUser = Security(requires_admin_user),
):
    """
    Cleanup ALL orphaned executions (>24h old) by directly updating DB status.
    Operates on all executions, not just paginated results.

    Returns:
        Number of executions cleaned up and success message
    """
    logger.info(f"Admin {user.user_id} cleaning up ALL orphaned executions")

    # Fetch all orphaned execution IDs
    execution_ids = await get_all_orphaned_execution_ids()

    if not execution_ids:
        return StopExecutionResponse(
            success=True,
            stopped_count=0,
            message="No orphaned executions to cleanup",
        )

    cleaned_count = await cleanup_orphaned_executions_bulk(execution_ids, user.user_id)

    return StopExecutionResponse(
        success=cleaned_count > 0,
        stopped_count=cleaned_count,
        message=f"Cleaned up {cleaned_count} orphaned executions",
    )


@router.post(
    "/diagnostics/executions/cleanup-all-stuck-queued",
    response_model=StopExecutionResponse,
    summary="Cleanup ALL Stuck Queued Executions",
)
async def cleanup_all_stuck_queued_executions_endpoint(
    user: AuthUser = Security(requires_admin_user),
):
    """
    Cleanup ALL stuck queued executions (QUEUED >1h) by updating DB status (admin only).
    Operates on entire dataset, not limited to pagination.

    Returns:
        Number of executions cleaned up and success message
    """
    logger.info(f"Admin {user.user_id} cleaning up ALL stuck queued executions")

    cleaned_count = await cleanup_all_stuck_queued_executions(user.user_id)

    return StopExecutionResponse(
        success=cleaned_count > 0,
        stopped_count=cleaned_count,
        message=f"Cleaned up {cleaned_count} stuck queued executions",
    )


@router.post(
    "/diagnostics/executions/requeue-all-stuck",
    response_model=RequeueExecutionResponse,
    summary="Requeue ALL Stuck Queued Executions",
)
async def requeue_all_stuck_executions(
    user: AuthUser = Security(requires_admin_user),
):
    """
    Requeue ALL stuck queued executions (QUEUED >1h) by publishing to RabbitMQ.
    Operates on all executions, not just paginated results.

    ⚠️ WARNING: This will re-execute ALL stuck executions and may cost significant credits.

    Returns:
        Number of executions requeued and success message
    """
    logger.info(f"Admin {user.user_id} requeueing ALL stuck queued executions")

    # Fetch all stuck queued execution IDs
    execution_ids = await get_all_stuck_queued_execution_ids()

    if not execution_ids:
        return RequeueExecutionResponse(
            success=True,
            requeued_count=0,
            message="No stuck queued executions to requeue",
        )

    requeued_count = await requeue_executions_bulk(execution_ids, user.user_id)

    return RequeueExecutionResponse(
        success=requeued_count > 0,
        requeued_count=requeued_count,
        message=f"Requeued {requeued_count} stuck executions",
    )
