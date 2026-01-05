"""
V2 External API - Schedules Endpoints

Provides endpoints for managing execution schedules.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.api.external.middleware import require_permission
from backend.data import graph as graph_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.user import get_user_by_id
from backend.executor import scheduler
from backend.util.clients import get_scheduler_client
from backend.util.timezone_utils import get_user_timezone_or_utc

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

logger = logging.getLogger(__name__)

schedules_router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class Schedule(BaseModel):
    """An execution schedule for a graph."""

    id: str
    name: str
    graph_id: str
    graph_version: int
    cron: str = Field(description="Cron expression for the schedule")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for scheduled executions"
    )
    next_run_time: Optional[datetime] = Field(
        default=None, description="Next scheduled run time"
    )
    is_enabled: bool = Field(default=True, description="Whether schedule is enabled")


class SchedulesListResponse(BaseModel):
    """Response for listing schedules."""

    schedules: list[Schedule]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class CreateScheduleRequest(BaseModel):
    """Request to create a schedule."""

    name: str = Field(description="Display name for the schedule")
    cron: str = Field(description="Cron expression (e.g., '0 9 * * *' for 9am daily)")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for scheduled executions"
    )
    credentials_inputs: dict[str, Any] = Field(
        default_factory=dict, description="Credentials for the schedule"
    )
    graph_version: Optional[int] = Field(
        default=None, description="Graph version (default: active version)"
    )
    timezone: Optional[str] = Field(
        default=None,
        description=(
            "Timezone for schedule (e.g., 'America/New_York'). "
            "Defaults to user's timezone."
        ),
    )


def _convert_schedule(job: scheduler.GraphExecutionJobInfo) -> Schedule:
    """Convert internal schedule job info to v2 API model."""
    # Parse the ISO format string to datetime
    next_run = datetime.fromisoformat(job.next_run_time) if job.next_run_time else None

    return Schedule(
        id=job.id,
        name=job.name or "",
        graph_id=job.graph_id,
        graph_version=job.graph_version,
        cron=job.cron,
        input_data=job.input_data,
        next_run_time=next_run,
        is_enabled=True,  # All returned schedules are enabled
    )


# ============================================================================
# Endpoints
# ============================================================================


@schedules_router.get(
    path="",
    summary="List all user schedules",
    response_model=SchedulesListResponse,
)
async def list_all_schedules(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_SCHEDULE)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> SchedulesListResponse:
    """
    List all schedules for the authenticated user across all graphs.
    """
    schedules = await get_scheduler_client().get_execution_schedules(
        user_id=auth.user_id
    )
    converted = [_convert_schedule(s) for s in schedules]

    # Manual pagination (scheduler doesn't support pagination natively)
    total_count = len(converted)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
    start = (page - 1) * page_size
    end = start + page_size
    paginated = converted[start:end]

    return SchedulesListResponse(
        schedules=paginated,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@schedules_router.delete(
    path="/{schedule_id}",
    summary="Delete a schedule",
)
async def delete_schedule(
    schedule_id: str = Path(description="Schedule ID to delete"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_SCHEDULE)
    ),
) -> None:
    """
    Delete an execution schedule.
    """
    try:
        await get_scheduler_client().delete_schedule(
            schedule_id=schedule_id,
            user_id=auth.user_id,
        )
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404, detail=f"Schedule #{schedule_id} not found"
            )
        raise


# ============================================================================
# Graph-specific Schedule Endpoints (nested under /graphs)
# These are included in the graphs router via include_router
# ============================================================================

graph_schedules_router = APIRouter()


@graph_schedules_router.get(
    path="/{graph_id}/schedules",
    summary="List schedules for a graph",
    response_model=list[Schedule],
)
async def list_graph_schedules(
    graph_id: str = Path(description="Graph ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_SCHEDULE)
    ),
) -> list[Schedule]:
    """
    List all schedules for a specific graph.
    """
    schedules = await get_scheduler_client().get_execution_schedules(
        user_id=auth.user_id,
        graph_id=graph_id,
    )
    return [_convert_schedule(s) for s in schedules]


@graph_schedules_router.post(
    path="/{graph_id}/schedules",
    summary="Create a schedule for a graph",
    response_model=Schedule,
)
async def create_graph_schedule(
    request: CreateScheduleRequest,
    graph_id: str = Path(description="Graph ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_SCHEDULE)
    ),
) -> Schedule:
    """
    Create a new execution schedule for a graph.

    The schedule will execute the graph at times matching the cron expression,
    using the provided input data.
    """
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=request.graph_version,
        user_id=auth.user_id,
    )
    if not graph:
        raise HTTPException(
            status_code=404,
            detail=f"Graph #{graph_id} v{request.graph_version} not found.",
        )

    # Determine timezone
    if request.timezone:
        user_timezone = request.timezone
    else:
        user = await get_user_by_id(auth.user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)

    result = await get_scheduler_client().add_execution_schedule(
        user_id=auth.user_id,
        graph_id=graph_id,
        graph_version=graph.version,
        name=request.name,
        cron=request.cron,
        input_data=request.input_data,
        input_credentials=request.credentials_inputs,
        user_timezone=user_timezone,
    )

    return _convert_schedule(result)
