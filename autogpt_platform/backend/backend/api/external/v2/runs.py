"""
V2 External API - Runs Endpoints

Provides access to agent runs and human-in-the-loop reviews.
"""

import logging
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission, ReviewStatus
from starlette import status

from backend.api.features.graph_executions import sharing
from backend.api.features.graph_executions.review.model import ReviewItem
from backend.api.features.graph_executions.review.service import process_reviews
from backend.data import execution as execution_db
from backend.data import human_review as review_db
from backend.data.execution import ExecutionStatus, GraphExecution
from backend.executor import utils as execution_utils

from .models import (
    AgentGraphRun,
    AgentGraphRunDetails,
    AgentRunReview,
    AgentRunReviewsSubmitRequest,
    AgentRunReviewsSubmitResponse,
    AgentRunReviewStatus,
    AgentRunShareResponse,
    RunStatus,
)
from .pagination import Page, PageRequest, page_request
from .tenancy import TenantContext, in_tenant, require_permission

logger = logging.getLogger(__name__)

runs_router = APIRouter(tags=["runs"])


# Registered before the `/{run_id}` routes below: Starlette matches in
# registration order, so `/{run_id}` would otherwise swallow `/reviews`.

# ============================================================================
# Endpoints - Reviews (Human-in-the-loop)
# ============================================================================


@runs_router.get(
    path="/reviews",
    summary="List agent run human-in-the-loop reviews",
    operation_id="listAgentRunReviews",
)
async def list_reviews(
    run_id: Optional[str] = Query(
        default=None, description="Filter by graph execution ID"
    ),
    review_status: Optional[AgentRunReviewStatus] = Query(
        default=None,
        alias="status",
        description="Filter by review status",
    ),
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(
        require_permission(APIKeyPermission.READ_RUN_REVIEW)
    ),
) -> Page[AgentRunReview]:
    """
    List human-in-the-loop reviews for agent runs.

    Returns reviews of all statuses if no status filter is given.
    """
    reviews, pagination = await review_db.get_reviews(
        user_id=auth.user_id,
        graph_exec_id=run_id,
        status=ReviewStatus(review_status) if review_status else None,
        page=page.page,
        page_size=page.limit,
        organization_id=auth.organization_id,
        graph_runs_only=True,
    )

    return page.paged(
        [AgentRunReview.from_internal(r) for r in reviews],
        total_count=pagination.total_items,
    )


@runs_router.post(
    path="/{run_id}/reviews",
    summary="Submit agent run human-in-the-loop reviews",
    operation_id="submitAgentRunReviews",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_reviews(
    request: AgentRunReviewsSubmitRequest,
    run_id: str = Path(description="Graph Execution ID"),
    auth: TenantContext = Security(
        require_permission(APIKeyPermission.WRITE_RUN_REVIEW)
    ),
) -> AgentRunReviewsSubmitResponse:
    """
    Submit responses to all pending human-in-the-loop reviews for a run.

    All pending reviews for the run must be included in the request.
    Approving a review continues execution; rejecting terminates that branch.
    """
    # Reviews carry no organization of their own; the run they belong to does.
    await _own_run(run_id, auth)

    outcome = await process_reviews(
        auth.user_id,
        [
            ReviewItem(
                node_exec_id=decision.node_exec_id,
                approved=decision.approved,
                reviewed_data=decision.edited_payload,
                message=decision.message,
                auto_approve_future=decision.auto_approve_future,
            )
            for decision in request.reviews
        ],
        graph_exec_id=run_id,
        organization_id=auth.organization_id,
        team_id=auth.team_id,
    )

    return AgentRunReviewsSubmitResponse(
        run_id=run_id,
        approved_count=outcome.approved_count,
        rejected_count=outcome.rejected_count,
    )


# ============================================================================
# Endpoints - Runs
# ============================================================================


@runs_router.get(
    path="",
    summary="List agent runs",
    operation_id="listAgentRuns",
)
async def list_runs(
    graph_id: Optional[str] = Query(default=None, description="Filter by graph ID"),
    # `Annotated`, not `= Query(...)`: the latter leaves the sentinel as the
    # Python default, and this handler is also called directly in tests.
    statuses: Annotated[
        Optional[list[RunStatus]],
        Query(description="Filter by run status; repeat to match several"),
    ] = None,
    started_after: Annotated[
        Optional[datetime], Query(description="Only runs created at or after this time")
    ] = None,
    started_before: Annotated[
        Optional[datetime],
        Query(description="Only runs created at or before this time"),
    ] = None,
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_RUN)),
) -> Page[AgentGraphRun]:
    """List agent runs, optionally filtered by graph, status and creation time."""
    result = await execution_db.get_graph_executions_paginated(
        user_id=auth.user_id,
        graph_id=graph_id,
        statuses=[ExecutionStatus(s) for s in statuses] if statuses else None,
        created_time_gte=started_after,
        created_time_lte=started_before,
        page=page.page,
        page_size=page.limit,
        organization_id=auth.organization_id,
    )

    return page.paged(
        [AgentGraphRun.from_internal(e) for e in result.executions],
        total_count=result.pagination.total_items,
    )


@runs_router.get(
    path="/{run_id}",
    summary="Get agent run details",
    operation_id="getAgentRunDetails",
)
async def get_run(
    run_id: str = Path(description="Graph Execution ID"),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_RUN)),
) -> AgentGraphRunDetails:
    """Get detailed information about a specific run."""
    result = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
        include_node_executions=True,
        organization_id=auth.organization_id,
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run #{run_id} not found",
        )

    return AgentGraphRunDetails.from_internal(result)


@runs_router.post(
    path="/{run_id}/stop",
    summary="Stop agent run",
    operation_id="stopAgentRun",
    status_code=status.HTTP_202_ACCEPTED,
)
async def stop_run(
    run_id: str = Path(description="Graph Execution ID"),
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_RUN)),
) -> AgentGraphRun:
    """
    Stop a run that hasn't finished: one that is incomplete, queued, running,
    or waiting for a review.

    Waits up to 15 seconds for the run to stop, then returns it with its status
    at that moment. A run that has already finished answers `409`.
    """
    run = await _own_run(run_id, auth)
    if run.status in _FINISHED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run #{run_id} has already finished ({run.status.value})",
        )

    try:
        await execution_utils.stop_graph_execution(
            graph_exec_id=run_id,
            user_id=auth.user_id,
            wait_timeout=_STOP_WAIT_SECONDS,
        )
    except TimeoutError:
        # The cancel is published; the executor is still winding the run
        # down, which the returned status shows.
        logger.warning(f"Run #{run_id} did not stop within {_STOP_WAIT_SECONDS}s")

    return AgentGraphRun.from_internal(await _own_run(run_id, auth))


@runs_router.delete(
    path="/{run_id}",
    summary="Delete agent run",
    operation_id="deleteAgentRun",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_run(
    run_id: str = Path(description="Graph Execution ID"),
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_RUN)),
) -> None:
    """Delete an agent run. A shared run stops being downloadable too."""
    await _own_run(run_id, auth)

    await sharing.delete_execution(auth.user_id, run_id)


# ============================================================================
# Endpoints - Sharing
# ============================================================================


@runs_router.post(
    path="/{run_id}/share",
    summary="Enable sharing for an agent run",
    operation_id="enableAgentRunShare",
    status_code=status.HTTP_201_CREATED,
)
async def enable_sharing(
    run_id: str = Path(description="Graph Execution ID"),
    auth: TenantContext = Security(
        require_permission(APIKeyPermission.READ_RUN, APIKeyPermission.SHARE_RUN)
    ),
) -> AgentRunShareResponse:
    """Enable public sharing for a run.

    Sharing again issues a new token, and links from the earlier share stop
    working.
    """
    await _own_run(run_id, auth)

    share_token = await sharing.share_execution(auth.user_id, run_id)

    return AgentRunShareResponse(
        share_url=sharing.share_url(share_token), share_token=share_token
    )


@runs_router.delete(
    path="/{run_id}/share",
    summary="Disable sharing for an agent run",
    operation_id="disableAgentRunShare",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def disable_sharing(
    run_id: str = Path(description="Graph Execution ID"),
    auth: TenantContext = Security(
        require_permission(APIKeyPermission.READ_RUN, APIKeyPermission.SHARE_RUN)
    ),
) -> None:
    """Disable public sharing for a run, and the file downloads it allowed."""
    await _own_run(run_id, auth)

    await sharing.unshare_execution(auth.user_id, run_id)


_FINISHED = frozenset(
    {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.TERMINATED}
)
_STOP_WAIT_SECONDS = 15.0


async def _own_run(run_id: str, auth: TenantContext) -> GraphExecution:
    """The caller's own run in this tenant, or 404, before acting on it.

    Reads may show a teammate's run in the same organization; stopping,
    deleting, sharing or reviewing it is that teammate's to do.
    """
    return in_tenant(
        await execution_db.get_graph_execution(
            user_id=auth.user_id, execution_id=run_id
        ),
        auth,
        f"Run #{run_id}",
    )
