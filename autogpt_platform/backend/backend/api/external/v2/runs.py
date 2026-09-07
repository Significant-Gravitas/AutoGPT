"""
V2 External API - Runs Endpoints

Provides access to agent runs and human-in-the-loop reviews.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission, ReviewStatus
from starlette import status

from backend.api.features.executions.review.model import ReviewItem
from backend.api.features.executions.review.service import process_reviews
from backend.data import execution as execution_db
from backend.data import human_review as review_db
from backend.executor import utils as execution_utils
from backend.util.settings import Settings

from .models import (
    AgentGraphRun,
    AgentGraphRunDetails,
    AgentRunReview,
    AgentRunReviewsSubmitRequest,
    AgentRunReviewsSubmitResponse,
    AgentRunReviewStatus,
    AgentRunShareResponse,
)
from .pagination import Page, PageRequest, page_request
from .tenancy import TenantContext, in_tenant, require_permission

logger = logging.getLogger(__name__)
settings = Settings()

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
    status: Optional[AgentRunReviewStatus] = Query(
        default=None,
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
        status=ReviewStatus(status) if status else None,
        page=page.page,
        page_size=page.limit,
        organization_id=auth.organization_id,
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
    await _assert_run_in_tenant(run_id, auth)

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
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_RUN)),
) -> Page[AgentGraphRun]:
    """List agent runs, optionally filtered by graph ID."""
    result = await execution_db.get_graph_executions_paginated(
        user_id=auth.user_id,
        graph_id=graph_id,
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
    Stop a running execution.

    Only runs with status QUEUED or RUNNING can be stopped.
    """
    # Verify the run exists and belongs to the user
    exec = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
        organization_id=auth.organization_id,
    )
    if not exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run #{run_id} not found",
        )

    # Stop the execution
    await execution_utils.stop_graph_execution(
        graph_exec_id=run_id,
        user_id=auth.user_id,
    )

    # Fetch updated execution
    updated_exec = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
        organization_id=auth.organization_id,
    )

    if not updated_exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run #{run_id} not found",
        )

    return AgentGraphRun.from_internal(updated_exec)


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
    """Delete an agent run."""
    await _assert_run_in_tenant(run_id, auth)

    await execution_db.delete_graph_execution(
        graph_exec_id=run_id,
        user_id=auth.user_id,
    )


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
    """Enable public sharing for a run."""
    execution = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
        organization_id=auth.organization_id,
    )
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run #{run_id} not found",
        )

    share_token = str(uuid.uuid4())

    await execution_db.update_graph_execution_share_status(
        execution_id=run_id,
        user_id=auth.user_id,
        is_shared=True,
        share_token=share_token,
        shared_at=datetime.now(timezone.utc),
    )

    frontend_url = settings.config.frontend_base_url or "http://localhost:3000"
    share_url = f"{frontend_url}/share/{share_token}"

    return AgentRunShareResponse(share_url=share_url, share_token=share_token)


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
    """Disable public sharing for a run."""
    execution = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
        organization_id=auth.organization_id,
    )
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run #{run_id} not found",
        )

    await execution_db.update_graph_execution_share_status(
        execution_id=run_id,
        user_id=auth.user_id,
        is_shared=False,
        share_token=None,
        shared_at=None,
    )


async def _assert_run_in_tenant(run_id: str, auth: TenantContext) -> None:
    """404 before acting on a run the credentials cannot reach."""
    in_tenant(
        await execution_db.get_graph_execution(
            user_id=auth.user_id,
            execution_id=run_id,
            organization_id=auth.organization_id,
        ),
        auth,
        f"Run #{run_id}",
    )
