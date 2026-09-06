"""
V2 External API - Schedules Endpoints

Provides endpoints for managing execution schedules.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from prisma.enums import APIKeyPermission
from starlette import status

from backend.data import graph as graph_db
from backend.data.tenancy import get_user_team_ids
from backend.data.user import get_user_by_id
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError
from backend.util.timezone_utils import get_user_timezone_or_utc

from .models import AgentRunSchedule, AgentRunScheduleCreateRequest
from .pagination import Page, PageRequest, page_request
from .tenancy import TenantContext, require_permission

logger = logging.getLogger(__name__)

schedules_router = APIRouter(tags=["graphs", "schedules"])


# ============================================================================
# Endpoints
# ============================================================================


@schedules_router.get(
    path="",
    summary="List run schedules",
    operation_id="listGraphRunSchedules",
)
async def list_all_schedules(
    graph_id: Optional[str] = Query(default=None, description="Filter by graph ID"),
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_SCHEDULE)),
) -> Page[AgentRunSchedule]:
    """List schedules for the authenticated user."""
    schedules = await get_scheduler_client().get_graph_execution_schedules(
        user_id=auth.user_id,
        graph_id=graph_id,
        organization_id=auth.organization_id,
        team_ids=await get_user_team_ids(auth.user_id, auth.organization_id),
    )
    # The scheduler keeps a schedule the caller owns even when it belongs to
    # another org, so the org filter is applied here rather than trusted there.
    return page.slice(
        [AgentRunSchedule.from_internal(s) for s in schedules if _in_tenant(s, auth)]
    )


@schedules_router.delete(
    path="/{schedule_id}",
    summary="Delete run schedule",
    operation_id="deleteGraphRunSchedule",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_schedule(
    schedule_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_SCHEDULE)),
) -> None:
    """Delete an execution schedule."""
    # The scheduler deletes by user id alone, so the org is checked here: a
    # schedule the key's org cannot see must not be deletable through it.
    await _assert_schedule_in_tenant(schedule_id, auth)

    try:
        await get_scheduler_client().delete_schedule(
            schedule_id=schedule_id,
            user_id=auth.user_id,
        )
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schedule #{schedule_id} not found",
            )
        raise


# ============================================================================
# Graph-specific Schedule Endpoints (nested under /graphs)
# These are included in the graphs router via include_router
# ============================================================================

graph_schedules_router = APIRouter(tags=["graphs"])


@graph_schedules_router.post(
    path="/{graph_id}/schedules",
    summary="Create run schedule",
    operation_id="createGraphRunSchedule",
    status_code=status.HTTP_201_CREATED,
)
async def create_graph_schedule(
    request: AgentRunScheduleCreateRequest,
    graph_id: str,
    auth: TenantContext = Security(
        require_permission(APIKeyPermission.WRITE_SCHEDULE, APIKeyPermission.RUN_AGENT)
    ),
) -> AgentRunSchedule:
    """Create a new execution schedule for a graph."""
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=request.graph_version,
        user_id=auth.user_id,
        organization_id=auth.organization_id,
    )
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
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
        input_data=request.inputs,
        input_credentials=request.credentials_inputs,
        user_timezone=user_timezone,
        organization_id=auth.organization_id,
        team_id=auth.team_id,
    )

    return AgentRunSchedule.from_internal(result)


async def _assert_schedule_in_tenant(schedule_id: str, auth: TenantContext) -> None:
    """404 for a schedule outside the organization the credentials act in."""
    visible = await get_scheduler_client().get_graph_execution_schedules(
        user_id=auth.user_id,
        organization_id=auth.organization_id,
        team_ids=await get_user_team_ids(auth.user_id, auth.organization_id),
    )
    if not any(
        schedule.id == schedule_id and _in_tenant(schedule, auth)
        for schedule in visible
    ):
        raise NotFoundError(f"Schedule #{schedule_id} not found")


def _in_tenant(schedule: GraphExecutionJobInfo, auth: TenantContext) -> bool:
    """Schedules store an untagged org as "", so normalise before comparing."""
    return (schedule.organization_id or None) in (None, auth.organization_id)
