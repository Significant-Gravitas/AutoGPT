import logging
from typing import List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status
from prisma.enums import ReviewStatus

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    get_graph_execution_meta,
    get_node_execution,
)
from backend.data.graph import get_graph_settings
from backend.data.human_review import (
    create_auto_approval_record,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    has_pending_reviews_for_graph_exec,
    process_all_reviews_for_execution,
)
from backend.executor.utils import add_graph_execution

from .model import PendingHumanReviewModel, ReviewRequest, ReviewResponse

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

    # Collect all node exec IDs from the request
    all_request_node_ids = {review.node_exec_id for review in request.reviews}

    if not all_request_node_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one review must be provided",
        )

    # Get graph execution ID from pending reviews to validate execution status
    all_pending = await get_pending_reviews_for_user(user_id)
    matching_review = next(
        (r for r in all_pending if r.node_exec_id in all_request_node_ids),
        None,
    )

    if not matching_review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending reviews found for the requested node executions",
        )

    graph_exec_id = matching_review.graph_exec_id

    # Validate execution status before processing reviews
    graph_exec_meta = await get_graph_execution_meta(
        user_id=user_id, execution_id=graph_exec_id
    )

    if not graph_exec_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph execution #{graph_exec_id} not found",
        )

    # Only allow processing reviews if execution is paused for review
    # or incomplete (partial execution with some reviews already processed)
    if graph_exec_meta.status not in (
        ExecutionStatus.REVIEW,
        ExecutionStatus.INCOMPLETE,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot process reviews while execution status is {graph_exec_meta.status}. "
            f"Reviews can only be processed when execution is paused (REVIEW status). "
            f"Current status: {graph_exec_meta.status}",
        )

    # Build review decisions map
    # Auto-approved reviews use original data (no modifications allowed)
    review_decisions = {}
    for review in request.reviews:
        review_status = (
            ReviewStatus.APPROVED if review.approved else ReviewStatus.REJECTED
        )
        reviewed_data = (
            None if request.auto_approve_future_actions else review.reviewed_data
        )
        review_decisions[review.node_exec_id] = (
            review_status,
            reviewed_data,
            review.message,
        )

    # Process all reviews
    updated_reviews = await process_all_reviews_for_execution(
        user_id=user_id,
        review_decisions=review_decisions,
    )

    # Create auto-approval records for approved reviews
    # Note: Processing sequentially to avoid event loop issues in tests
    if request.auto_approve_future_actions:
        for node_exec_id, review in updated_reviews.items():
            if review.status == ReviewStatus.APPROVED:
                try:
                    node_exec = await get_node_execution(node_exec_id)
                    if node_exec:
                        await create_auto_approval_record(
                            user_id=user_id,
                            graph_exec_id=review.graph_exec_id,
                            graph_id=review.graph_id,
                            graph_version=review.graph_version,
                            node_id=node_exec.node_id,
                            payload=review.payload,
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to create auto-approval record for {node_exec_id}",
                        exc_info=e,
                    )

    # Count results
    approved_count = sum(
        1
        for review in updated_reviews.values()
        if review.status == ReviewStatus.APPROVED
    )
    rejected_count = sum(
        1
        for review in updated_reviews.values()
        if review.status == ReviewStatus.REJECTED
    )

    # Resume execution only if ALL pending reviews for this execution have been processed
    if updated_reviews:
        still_has_pending = await has_pending_reviews_for_graph_exec(graph_exec_id)

        if not still_has_pending:
            # Get the graph_id from any processed review
            first_review = next(iter(updated_reviews.values()))

            try:
                settings = await get_graph_settings(
                    user_id=user_id, graph_id=first_review.graph_id
                )

                execution_context = ExecutionContext(
                    human_in_the_loop_safe_mode=settings.human_in_the_loop_safe_mode,
                    sensitive_action_safe_mode=settings.sensitive_action_safe_mode,
                )

                await add_graph_execution(
                    graph_id=first_review.graph_id,
                    user_id=user_id,
                    graph_exec_id=graph_exec_id,
                    execution_context=execution_context,
                )
                logger.info(f"Resumed execution {graph_exec_id}")
            except Exception as e:
                logger.error(f"Failed to resume execution {graph_exec_id}: {str(e)}")

    return ReviewResponse(
        approved_count=approved_count,
        rejected_count=rejected_count,
        failed_count=0,
        error=None,
    )
