"""
V2 External API - Runs Endpoints

Provides access to execution runs and human-in-the-loop reviews.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission, ReviewStatus
from pydantic import JsonValue
from starlette.status import HTTP_204_NO_CONTENT

from backend.api.external.middleware import require_permission
from backend.data import execution as execution_db
from backend.data import human_review as review_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.executor import utils as execution_utils
from backend.util.settings import Settings

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    AgentGraphRun,
    AgentGraphRunDetails,
    AgentRunListResponse,
    AgentRunReview,
    AgentRunReviewsResponse,
    AgentRunReviewsSubmitRequest,
    AgentRunReviewsSubmitResponse,
    AgentRunShareResponse,
)

logger = logging.getLogger(__name__)
settings = Settings()

runs_router = APIRouter()


# ============================================================================
# Endpoints - Runs
# ============================================================================


@runs_router.get(
    path="",
    summary="List agent runs",
)
async def list_runs(
    graph_id: Optional[str] = Query(default=None, description="Filter by graph ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_RUN)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> AgentRunListResponse:
    """
    List agent runs for the authenticated user.

    Optionally filter by graph ID.
    """
    result = await execution_db.get_graph_executions_paginated(
        user_id=auth.user_id,
        graph_id=graph_id,
        page=page,
        page_size=page_size,
    )

    return AgentRunListResponse(
        runs=[AgentGraphRun.from_internal(e) for e in result.executions],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@runs_router.get(
    path="/{run_id}",
    summary="Get run details",
)
async def get_run(
    run_id: str = Path(description="Graph Execution ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_RUN)
    ),
) -> AgentGraphRunDetails:
    """
    Get detailed information about a specific run.

    Includes outputs and individual node execution results.
    """
    result = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
        include_node_executions=True,
    )

    if not result:
        raise HTTPException(status_code=404, detail=f"Run #{run_id} not found")

    return AgentGraphRunDetails.from_internal(result)


@runs_router.post(
    path="/{run_id}/stop",
    summary="Stop a run",
)
async def stop_run(
    run_id: str = Path(description="Graph Execution ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_RUN)
    ),
) -> AgentGraphRun:
    """
    Stop a running execution.

    Only runs in QUEUED or RUNNING status can be stopped.
    """
    # Verify the run exists and belongs to the user
    exec = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
    )
    if not exec:
        raise HTTPException(status_code=404, detail=f"Run #{run_id} not found")

    # Stop the execution
    await execution_utils.stop_graph_execution(
        graph_exec_id=run_id,
        user_id=auth.user_id,
    )

    # Fetch updated execution
    updated_exec = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
    )

    if not updated_exec:
        raise HTTPException(status_code=404, detail=f"Run #{run_id} not found")

    return AgentGraphRun.from_internal(updated_exec)


@runs_router.delete(
    path="/{run_id}",
    summary="Delete a run",
)
async def delete_run(
    run_id: str = Path(description="Graph Execution ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_RUN)
    ),
) -> None:
    """
    Delete an execution run.

    This marks the run as deleted. The data may still be retained for
    some time for recovery purposes.
    """
    await execution_db.delete_graph_execution(
        graph_exec_id=run_id,
        user_id=auth.user_id,
    )


# ============================================================================
# Endpoints - Sharing
# ============================================================================


@runs_router.post(
    path="/{run_id}/share",
    summary="Enable sharing for a run",
)
async def enable_sharing(
    run_id: str = Path(description="Graph Execution ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_RUN, APIKeyPermission.SHARE_RUN)
    ),
) -> AgentRunShareResponse:
    """
    Enable public sharing for a run.

    Returns a public URL and share token that can be used to view the run
    without authentication.
    """
    execution = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
    )
    if not execution:
        raise HTTPException(status_code=404, detail=f"Run #{run_id} not found")

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
    summary="Disable sharing for a run",
    status_code=HTTP_204_NO_CONTENT,
)
async def disable_sharing(
    run_id: str = Path(description="Graph Execution ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.SHARE_RUN)
    ),
) -> None:
    """
    Disable public sharing for a run.

    The share URL will no longer work after this call.
    """
    execution = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=run_id,
    )
    if not execution:
        raise HTTPException(status_code=404, detail=f"Run #{run_id} not found")

    await execution_db.update_graph_execution_share_status(
        execution_id=run_id,
        user_id=auth.user_id,
        is_shared=False,
        share_token=None,
        shared_at=None,
    )


# ============================================================================
# Endpoints - Reviews (Human-in-the-loop)
# ============================================================================


@runs_router.get(
    path="/reviews",
    summary="List human-in-the-loop reviews for agent runs",
)
async def list_reviews(
    run_id: Optional[str] = Query(default=None, description="Filter by run ID"),
    status: Optional[ReviewStatus] = Query(
        description="Filter by review status",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_RUN_REVIEW)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> AgentRunReviewsResponse:
    """
    List human-in-the-loop reviews.

    Defaults to pending (WAITING) reviews. Use `status` to filter by
    review status and `run_id` to scope to a specific run.
    """
    reviews, pagination = await review_db.get_reviews(
        user_id=auth.user_id,
        graph_exec_id=run_id,
        status=status,
        page=page,
        page_size=page_size,
    )

    return AgentRunReviewsResponse(
        reviews=[AgentRunReview.from_internal(r) for r in reviews],
        page=pagination.current_page,
        page_size=pagination.page_size,
        total_count=pagination.total_items,
        total_pages=pagination.total_pages,
    )


@runs_router.post(
    path="/{run_id}/reviews",
    summary="Submit a human-in-the-loop review for an agent run",
)
async def submit_reviews(
    request: AgentRunReviewsSubmitRequest,
    run_id: str = Path(description="Graph Execution ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_RUN_REVIEW)
    ),
) -> AgentRunReviewsSubmitResponse:
    """
    Submit responses to all pending human-in-the-loop reviews for an agent run.

    All pending reviews for the run must be included. Approving
    a review will allow the agent to continue; rejecting will terminate
    execution at that point.
    """
    # Build review decisions dict for process_all_reviews_for_execution
    review_decisions: dict[str, tuple[ReviewStatus, JsonValue | None, str | None]] = {}

    for decision in request.reviews:
        status = ReviewStatus.APPROVED if decision.approved else ReviewStatus.REJECTED
        review_decisions[decision.node_exec_id] = (
            status,
            decision.edited_payload,
            decision.message,
        )

    results = await review_db.process_all_reviews_for_execution(
        user_id=auth.user_id,
        review_decisions=review_decisions,
    )

    approved_count = sum(
        1 for r in results.values() if r.status == ReviewStatus.APPROVED
    )
    rejected_count = sum(
        1 for r in results.values() if r.status == ReviewStatus.REJECTED
    )

    return AgentRunReviewsSubmitResponse(
        run_id=run_id,
        approved_count=approved_count,
        rejected_count=rejected_count,
    )
