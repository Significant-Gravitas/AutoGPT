"""
V2 External API - Runs Endpoints

Provides access to execution runs and human-in-the-loop reviews.
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission, ReviewStatus
from pydantic import JsonValue

from backend.api.external.middleware import require_permission
from backend.data import execution as execution_db
from backend.data import human_review as review_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.executor import utils as execution_utils

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    PendingReview,
    PendingReviewsResponse,
    Run,
    RunDetails,
    RunsListResponse,
    SubmitReviewsRequest,
    SubmitReviewsResponse,
)

logger = logging.getLogger(__name__)

runs_router = APIRouter()


# ============================================================================
# Endpoints - Runs
# ============================================================================


@runs_router.get(
    path="",
    summary="List all runs",
    response_model=RunsListResponse,
)
async def list_runs(
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
) -> RunsListResponse:
    """
    List all execution runs for the authenticated user.

    Returns runs across all agents, sorted by most recent first.
    """
    result = await execution_db.get_graph_executions_paginated(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    return RunsListResponse(
        runs=[Run.from_internal(e) for e in result.executions],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_pages=result.pagination.total_pages,
    )


@runs_router.get(
    path="/{run_id}",
    summary="Get run details",
    response_model=RunDetails,
)
async def get_run(
    run_id: str = Path(description="Run ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_RUN)
    ),
) -> RunDetails:
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

    return RunDetails.from_internal(result)


@runs_router.post(
    path="/{run_id}/stop",
    summary="Stop a run",
)
async def stop_run(
    run_id: str = Path(description="Run ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_RUN)
    ),
) -> Run:
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

    return Run.from_internal(updated_exec)


@runs_router.delete(
    path="/{run_id}",
    summary="Delete a run",
)
async def delete_run(
    run_id: str = Path(description="Run ID"),
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
# Endpoints - Reviews (Human-in-the-loop)
# ============================================================================


@runs_router.get(
    path="/reviews",
    summary="List all pending reviews",
    response_model=PendingReviewsResponse,
)
async def list_pending_reviews(
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
) -> PendingReviewsResponse:
    """
    List all pending human-in-the-loop reviews.

    These are blocks that require human approval or input before the
    agent can continue execution.
    """
    reviews = await review_db.get_pending_reviews_for_user(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    # Note: get_pending_reviews_for_user returns list directly, not a paginated result
    # We compute pagination info based on results
    total_count = len(reviews)
    total_pages = max(1, (total_count + page_size - 1) // page_size)

    return PendingReviewsResponse(
        reviews=[PendingReview.from_internal(r) for r in reviews],
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@runs_router.get(
    path="/{run_id}/reviews",
    summary="List reviews for a run",
    response_model=list[PendingReview],
)
async def list_run_reviews(
    run_id: str = Path(description="Run ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_RUN_REVIEW)
    ),
) -> list[PendingReview]:
    """
    List all human-in-the-loop reviews for a specific run.
    """
    reviews = await review_db.get_pending_reviews_for_execution(
        graph_exec_id=run_id,
        user_id=auth.user_id,
    )

    return [PendingReview.from_internal(r) for r in reviews]


@runs_router.post(
    path="/{run_id}/reviews",
    summary="Submit review responses for a run",
    response_model=SubmitReviewsResponse,
)
async def submit_reviews(
    request: SubmitReviewsRequest,
    run_id: str = Path(description="Run ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_RUN_REVIEW)
    ),
) -> SubmitReviewsResponse:
    """
    Submit responses to all pending human-in-the-loop reviews for a run.

    All pending reviews for the execution must be included. Approving
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

    try:
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

        return SubmitReviewsResponse(
            run_id=run_id,
            approved_count=approved_count,
            rejected_count=rejected_count,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
