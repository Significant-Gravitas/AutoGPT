import logging
from typing import List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status

from backend.copilot.constants import is_copilot_synthetic_id
from backend.data.execution import get_graph_execution_meta
from backend.data.human_review import (
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
)

from .model import PendingHumanReviewModel, ReviewRequest, ReviewResponse
from .service import process_reviews

logger = logging.getLogger(__name__)


router = APIRouter(
    tags=["v2", "executions", "review"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "/pending",
    summary="Get Pending Reviews",
    response_model=List[PendingHumanReviewModel],
    responses={
        200: {"description": "List of pending reviews"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(25, ge=1, le=100, description="Number of reviews per page"),
) -> List[PendingHumanReviewModel]:
    """Get all pending reviews for the current user.

    Retrieves all reviews with status "WAITING" that belong to the authenticated user.
    Results are ordered by creation time (newest first).

    Args:
        user_id: Authenticated user ID from security dependency

    Returns:
        List of pending review objects with status converted to typed literals

    Raises:
        HTTPException: If authentication fails or database error occurs

    Note:
        Reviews with invalid status values are logged as warnings but excluded
        from results rather than failing the entire request.
    """

    return await get_pending_reviews_for_user(user_id, page, page_size)


@router.get(
    "/execution/{graph_exec_id}",
    summary="Get Pending Reviews for Execution",
    response_model=List[PendingHumanReviewModel],
    responses={
        200: {"description": "List of pending reviews for the execution"},
        404: {"description": "Graph execution not found"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews_for_execution(
    graph_exec_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> List[PendingHumanReviewModel]:
    """Get all pending reviews for a specific graph execution.

    Retrieves all reviews with status "WAITING" for the specified graph execution
    that belong to the authenticated user. Results are ordered by creation time
    (oldest first) to preserve review order within the execution.

    Args:
        graph_exec_id: ID of the graph execution to get reviews for
        user_id: Authenticated user ID from security dependency

    Returns:
        List of pending review objects for the specified execution

    Raises:
        HTTPException:
            - 404: If the graph execution doesn't exist or isn't owned by this user
            - 500: If authentication fails or database error occurs

    Note:
        Only returns reviews owned by the authenticated user for security.
        Reviews with invalid status are excluded with warning logs.
    """

    # Verify user owns the graph execution before returning reviews
    # (CoPilot synthetic IDs don't have graph execution records)
    if not is_copilot_synthetic_id(graph_exec_id):
        graph_exec = await get_graph_execution_meta(
            user_id=user_id, execution_id=graph_exec_id
        )
        if not graph_exec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph execution #{graph_exec_id} not found",
            )

    return await get_pending_reviews_for_execution(graph_exec_id, user_id)


@router.post("/action", response_model=ReviewResponse)
async def process_review_action(
    request: ReviewRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ReviewResponse:
    """Process reviews with approve or reject actions."""
    outcome = await process_reviews(user_id, request.reviews)

    error_message = None
    if outcome.auto_approval_failed_count:
        error_message = (
            f"{outcome.auto_approval_failed_count} auto-approval setting(s) could "
            f"not be saved. You may need to manually approve these reviews in "
            f"future executions."
        )

    return ReviewResponse(
        approved_count=outcome.approved_count,
        rejected_count=outcome.rejected_count,
        failed_count=outcome.auto_approval_failed_count,
        error=error_message,
    )
