import asyncio
import logging
from typing import Any, List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status
from prisma.enums import ReviewStatus

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    get_graph_execution_meta,
)
from backend.data.graph import get_graph_settings
from backend.data.human_review import (
    create_auto_approval_record,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    get_reviews_by_node_exec_ids,
    has_pending_reviews_for_graph_exec,
    process_all_reviews_for_execution,
)
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.data.user import get_user_by_id
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

    # Batch fetch all requested reviews (regardless of status for idempotent handling)
    reviews_map = await get_reviews_by_node_exec_ids(
        list(all_request_node_ids), user_id
    )

    # Validate all reviews were found (must exist, any status is OK for now)
    missing_ids = all_request_node_ids - set(reviews_map.keys())
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review(s) not found: {', '.join(missing_ids)}",
        )

    # Validate all reviews belong to the same execution
    graph_exec_ids = {review.graph_exec_id for review in reviews_map.values()}
    if len(graph_exec_ids) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="All reviews in a single request must belong to the same execution.",
        )

    graph_exec_id = next(iter(graph_exec_ids))

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

    # Build review decisions map and track which reviews requested auto-approval
    # Auto-approved reviews use original data (no modifications allowed)
    review_decisions = {}
    auto_approve_requests = {}  # Map node_exec_id -> auto_approve_future flag

    for review in request.reviews:
        review_status = (
            ReviewStatus.APPROVED if review.approved else ReviewStatus.REJECTED
        )
        # If this review requested auto-approval, don't allow data modifications
        reviewed_data = None if review.auto_approve_future else review.reviewed_data
        review_decisions[review.node_exec_id] = (
            review_status,
            reviewed_data,
            review.message,
        )
        auto_approve_requests[review.node_exec_id] = review.auto_approve_future

    # Process all reviews
    updated_reviews = await process_all_reviews_for_execution(
        user_id=user_id,
        review_decisions=review_decisions,
    )

    # Create auto-approval records for approved reviews that requested it
    # Deduplicate by node_id to avoid race conditions when multiple reviews
    # for the same node are processed in parallel
    async def create_auto_approval_for_node(
        node_id: str, review_result
    ) -> tuple[str, bool]:
        """
        Create auto-approval record for a node.
        Returns (node_id, success) tuple for tracking failures.
        """
        try:
            await create_auto_approval_record(
                user_id=user_id,
                graph_exec_id=review_result.graph_exec_id,
                graph_id=review_result.graph_id,
                graph_version=review_result.graph_version,
                node_id=node_id,
                payload=review_result.payload,
            )
            return (node_id, True)
        except Exception as e:
            logger.error(
                f"Failed to create auto-approval record for node {node_id}",
                exc_info=e,
            )
            return (node_id, False)

    # Collect node_exec_ids that need auto-approval
    node_exec_ids_needing_auto_approval = [
        node_exec_id
        for node_exec_id, review_result in updated_reviews.items()
        if review_result.status == ReviewStatus.APPROVED
        and auto_approve_requests.get(node_exec_id, False)
    ]

    # Batch-fetch node executions to get node_ids
    nodes_needing_auto_approval: dict[str, Any] = {}
    if node_exec_ids_needing_auto_approval:
        from backend.data.execution import get_node_executions

        node_execs = await get_node_executions(
            graph_exec_id=graph_exec_id, include_exec_data=False
        )
        node_exec_map = {node_exec.node_exec_id: node_exec for node_exec in node_execs}

        for node_exec_id in node_exec_ids_needing_auto_approval:
            node_exec = node_exec_map.get(node_exec_id)
            if node_exec:
                review_result = updated_reviews[node_exec_id]
                # Use the first approved review for this node (deduplicate by node_id)
                if node_exec.node_id not in nodes_needing_auto_approval:
                    nodes_needing_auto_approval[node_exec.node_id] = review_result
            else:
                logger.error(
                    f"Failed to create auto-approval record for {node_exec_id}: "
                    f"Node execution not found. This may indicate a race condition "
                    f"or data inconsistency."
                )

    # Execute all auto-approval creations in parallel (deduplicated by node_id)
    auto_approval_results = await asyncio.gather(
        *[
            create_auto_approval_for_node(node_id, review_result)
            for node_id, review_result in nodes_needing_auto_approval.items()
        ],
        return_exceptions=True,
    )

    # Count auto-approval failures
    auto_approval_failed_count = 0
    for result in auto_approval_results:
        if isinstance(result, Exception):
            # Unexpected exception during auto-approval creation
            auto_approval_failed_count += 1
            logger.error(
                f"Unexpected exception during auto-approval creation: {result}"
            )
        elif isinstance(result, tuple) and len(result) == 2 and not result[1]:
            # Auto-approval creation failed (returned False)
            auto_approval_failed_count += 1

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
                # Fetch user and settings to build complete execution context
                user = await get_user_by_id(user_id)
                settings = await get_graph_settings(
                    user_id=user_id, graph_id=first_review.graph_id
                )

                # Preserve user's timezone preference when resuming execution
                user_timezone = (
                    user.timezone if user.timezone != USER_TIMEZONE_NOT_SET else "UTC"
                )

                execution_context = ExecutionContext(
                    human_in_the_loop_safe_mode=settings.human_in_the_loop_safe_mode,
                    sensitive_action_safe_mode=settings.sensitive_action_safe_mode,
                    user_timezone=user_timezone,
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

    # Build error message if auto-approvals failed
    error_message = None
    if auto_approval_failed_count > 0:
        error_message = (
            f"{auto_approval_failed_count} auto-approval setting(s) could not be saved. "
            f"You may need to manually approve these reviews in future executions."
        )

    return ReviewResponse(
        approved_count=approved_count,
        rejected_count=rejected_count,
        failed_count=auto_approval_failed_count,
        error=error_message,
    )
