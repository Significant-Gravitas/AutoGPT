import logging
from typing import List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status

from backend.data.execution import (
    ExecutionStatus,
    get_graph_execution_meta,
    update_graph_execution_stats,
)
from backend.data.human_review import (
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    update_review_action,
)
from backend.server.v2.executions.review.model import (
    PendingHumanReviewModel,
    ReviewActionRequest,
    ReviewActionResponse,
)

logger = logging.getLogger(__name__)


def handle_database_error(
    operation: str, resource_id: str, error: Exception
) -> HTTPException:
    """Centralized database error handling for review operations."""
    logger.error(f"Database error during {operation} for {resource_id}: {str(error)}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal server error during {operation}",
    )


router = APIRouter(
    prefix="/review",
    tags=["executions", "review", "private"],
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
        400: {"description": "Invalid graph execution ID"},
        403: {"description": "Access denied to graph execution"},
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
            - 403: If user doesn't own the graph execution
            - 500: If authentication fails or database error occurs

    Note:
        Only returns reviews owned by the authenticated user for security.
        Reviews with invalid status are excluded with warning logs.
    """

    # Verify user owns the graph execution before returning reviews
    try:
        graph_exec = await get_graph_execution_meta(
            user_id=user_id, execution_id=graph_exec_id
        )
        if not graph_exec:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to graph execution",
            )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise handle_database_error("graph ownership verification", graph_exec_id, e)

    return await get_pending_reviews_for_execution(graph_exec_id, user_id)


@router.post(
    "/{node_exec_id}/action",
    summary="Process Review Action",
    response_model=ReviewActionResponse,
    responses={
        200: {"description": "Review action processed successfully"},
        400: {"description": "Invalid request or review already processed"},
        403: {"description": "Access denied"},
        404: {"description": "Review not found"},
        422: {"description": "Validation error"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def review_data(
    node_exec_id: str,
    request: ReviewActionRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ReviewActionResponse:
    """Approve or reject pending review data.

    Processes a human review action (approve/reject) for a pending review.
    Includes comprehensive security checks and atomic database operations.

    Security Features:
    - Verifies review ownership by authenticated user
    - Validates graph execution access permissions
    - Prevents modification of non-WAITING reviews
    - Uses database transactions for atomic updates

    Args:
        node_exec_id: Node execution ID of the review to process
        request: Review action details (approve/reject + optional data/message)
        user_id: Authenticated user ID from security dependency

    Returns:
        Success response with action status

    Raises:
        HTTPException:
            - 404: Review not found
            - 403: Access denied to review or graph execution
            - 400: Review already processed (not WAITING)
            - 500: Database or internal server errors

    Note:
        On approval, optionally modified data is stored and graph execution
        is resumed automatically. On rejection, execution remains paused.
        All operations use database transactions for consistency.
    """

    try:
        # Update the review using the data layer
        updated_review = await update_review_action(
            node_exec_id=node_exec_id,
            user_id=user_id,
            action=request.action,
            reviewed_data=request.reviewed_data,
            message=request.message,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error updating review {node_exec_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing review",
        )

    if updated_review is None:
        # Could be not found, access denied, already processed, or editing not allowed
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review not found, access denied, already processed, or editing not allowed",
        )

    # If approved, trigger graph execution resume
    if request.action == "approve":
        await _resume_graph_execution(updated_review.graph_exec_id)

    return ReviewActionResponse(action=request.action)


async def _resume_graph_execution(graph_exec_id: str) -> None:
    """Resume a graph execution by updating its status to QUEUED.

    Updates the graph execution status to QUEUED so the scheduler will
    pick it up for continued execution after human review approval.

    Args:
        graph_exec_id: Unique identifier of the graph execution to resume

    Raises:
        ValueError: If graph execution is not found
        Exception: For database connection or update errors

    Note:
        This function updates the graph execution status to resume processing
        after a human review has been completed.
    """
    try:
        # Update the graph execution status to QUEUED so the scheduler picks it up
        # Note: The graph execution is guaranteed to exist due to cascade relationship
        # with PendingHumanReview, so we can directly update the status
        await update_graph_execution_stats(
            graph_exec_id=graph_exec_id, status=ExecutionStatus.QUEUED
        )

        logger.info(f"Resumed graph execution {graph_exec_id}")

    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Failed to resume graph execution {graph_exec_id}: {e}")
        raise  # Re-raise to ensure calling code is aware of the failure
