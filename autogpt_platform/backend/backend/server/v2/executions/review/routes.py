import logging
from datetime import datetime, timezone
from typing import List, Literal

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Security, status
from prisma.enums import ReviewStatus
from prisma.errors import RecordNotFoundError
from prisma.models import PendingHumanReview

from backend.data.db import transaction
from backend.data.execution import ExecutionStatus, update_graph_execution_stats
from backend.server.v2.executions.review.model import (
    PendingHumanReviewResponse,
    ReviewActionRequest,
    ReviewActionResponse,
)
from backend.util.clients import get_database_manager_async_client
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


def _convert_review_status(status) -> Literal["WAITING", "APPROVED", "REJECTED"]:
    """Convert database enum status to typed literal for API response.

    Args:
        status: Database status value (enum or string)

    Returns:
        Validated status as typed literal

    Raises:
        ValueError: If status is not a valid review status

    Note:
        Handles both enum values (from new schema) and string values
        (for backward compatibility during migration).
    """
    # Handle both enum and string cases (for migration compatibility)
    status_str = status.value if hasattr(status, "value") else str(status)
    valid_statuses = {"WAITING", "APPROVED", "REJECTED"}
    if status_str not in valid_statuses:
        logger.warning(f"Invalid review status found in database: {status_str}")
        raise ValueError(f"Invalid review status: {status_str}")
    return status_str  # type: ignore


router = APIRouter(
    prefix="/review",
    tags=["execution-review", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "/pending",
    summary="Get Pending Reviews",
    responses={
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> List[PendingHumanReviewResponse]:
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

    reviews = await PendingHumanReview.prisma().find_many(
        where={"userId": user_id, "status": ReviewStatus.WAITING},
        order={"createdAt": "desc"},
    )

    result = []
    for review in reviews:
        try:
            converted_status = _convert_review_status(review.status)
            result.append(
                PendingHumanReviewResponse(
                    id=review.id,
                    user_id=review.userId,
                    node_exec_id=review.nodeExecId,
                    graph_exec_id=review.graphExecId,
                    graph_id=review.graphId,
                    graph_version=review.graphVersion,
                    data=review.data,
                    status=converted_status,
                    review_message=review.reviewMessage,
                    was_edited=review.wasEdited,
                    created_at=review.createdAt,
                    updated_at=review.updatedAt,
                    reviewed_at=review.reviewedAt,
                )
            )
        except ValueError as e:
            logger.error(f"Skipping review {review.id} due to invalid status: {e}")
            continue
    return result


@router.get(
    "/execution/{graph_exec_id}",
    summary="Get Pending Reviews for Execution",
    responses={
        403: {"description": "Access denied to graph execution"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews_for_execution(
    graph_exec_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> List[PendingHumanReviewResponse]:
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
    db = get_database_manager_async_client()
    try:
        graph_exec = await db.get_graph_execution_meta(
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
        logger.error(
            f"Database error while verifying graph ownership for execution {graph_exec_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while verifying access",
        )

    reviews = await PendingHumanReview.prisma().find_many(
        where={
            "userId": user_id,
            "graphExecId": graph_exec_id,
            "status": ReviewStatus.WAITING,
        },
        order={"createdAt": "asc"},
    )

    result = []
    for review in reviews:
        try:
            converted_status = _convert_review_status(review.status)
            result.append(
                PendingHumanReviewResponse(
                    id=review.id,
                    user_id=review.userId,
                    node_exec_id=review.nodeExecId,
                    graph_exec_id=review.graphExecId,
                    graph_id=review.graphId,
                    graph_version=review.graphVersion,
                    data=review.data,
                    status=converted_status,
                    review_message=review.reviewMessage,
                    was_edited=review.wasEdited,
                    created_at=review.createdAt,
                    updated_at=review.updatedAt,
                    reviewed_at=review.reviewedAt,
                )
            )
        except ValueError as e:
            logger.error(f"Skipping review {review.id} due to invalid status: {e}")
            continue
    return result


@router.post(
    "/{review_id}/action",
    summary="Review Data",
    responses={
        200: {"description": "Success", "content": {"application/json": {}}},
        400: {"description": "Review already processed"},
        403: {"description": "Access denied"},
        404: {"description": "Review not found"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def review_data(
    review_id: str,
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
        review_id: Unique identifier of the review to process
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

    # Find the review and verify ownership
    review = await PendingHumanReview.prisma().find_unique(where={"id": review_id})

    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Review not found"
        )

    if review.userId != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this review"
        )

    # Additional security check: verify user owns the graph execution
    db = get_database_manager_async_client()
    try:
        graph_exec = await db.get_graph_execution_meta(
            user_id=user_id, execution_id=review.graphExecId
        )
        if not graph_exec:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to graph execution",
            )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(
            f"Database error while verifying graph ownership for review {review_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while verifying access",
        )

    if review.status != ReviewStatus.WAITING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Review is already {review.status.value.lower()}",
        )

    # Update the review
    now = datetime.now(timezone.utc)
    update_status = (
        ReviewStatus.APPROVED if request.action == "approve" else ReviewStatus.REJECTED
    )

    # Handle reviewed_data for approve action and determine if data was edited
    review_data = review.data
    was_edited = False

    if request.action == "approve" and request.reviewed_data is not None:
        # Check if editing is allowed based on the editable flag
        editable = False
        if isinstance(review.data, dict):
            editable = review.data.get("editable", False)

        if not editable:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data modification not allowed - this review is read-only",
            )

        # Check if the data was actually modified
        original_data = (
            review.data.get("data")
            if isinstance(review.data, dict) and "data" in review.data
            else review.data
        )
        was_edited = original_data != request.reviewed_data

        # Update only the data part while preserving the structure
        if isinstance(review.data, dict) and "data" in review.data:
            review_data = {
                **review.data,
                "data": request.reviewed_data,  # Update just the data field
            }
        else:
            # Fallback: replace entire data
            review_data = request.reviewed_data

    # Use database transaction for atomic update and status change

    async with transaction() as tx:
        # Update the review atomically
        await tx.pendinghumanreview.update(
            where={"id": review_id},
            data={
                "status": update_status,
                "data": SafeJson(review_data),  # Store the (possibly modified) data
                "reviewMessage": request.message,
                "wasEdited": was_edited if request.action == "approve" else None,
                "reviewedAt": now,
            },
        )

        # If approved, trigger graph execution resume within the same transaction
        if request.action == "approve":
            await _resume_graph_execution_atomic(review.graphExecId, tx)

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
        This is a non-atomic helper function. For transactional operations,
        use _resume_graph_execution_atomic instead.
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


async def _resume_graph_execution_atomic(graph_exec_id: str, tx) -> None:
    """Resume a graph execution by updating its status within a database transaction.

    Atomically updates the graph execution status to QUEUED within an existing
    database transaction. This ensures consistency with other transaction operations.

    Args:
        graph_exec_id: Unique identifier of the graph execution to resume
        tx: Active database transaction context from Prisma

    Raises:
        ValueError: If graph execution is not found
        Exception: For database update errors (will rollback transaction)

    Note:
        This function is designed to be called within a database transaction.
        Any exceptions will cause transaction rollback to maintain consistency.
    """
    try:

        logger.info(f"Resuming graph execution {graph_exec_id} atomically")

        # Update the graph status to QUEUED to restart execution within transaction
        try:
            await tx.agentgraphexecution.update(
                where={"id": graph_exec_id},
                data={"executionStatus": ExecutionStatus.QUEUED.value},
            )
        except RecordNotFoundError:
            logger.error(
                f"Graph execution {graph_exec_id} not found during atomic update"
            )
            raise ValueError(f"Graph execution {graph_exec_id} not found")

        logger.info(
            f"Graph execution {graph_exec_id} status updated to QUEUED atomically"
        )

    except ValueError:
        raise  # Re-raise validation errors to rollback transaction
    except Exception as e:
        logger.error(
            f"Failed to resume graph execution {graph_exec_id} atomically: {str(e)}"
        )
        raise  # Re-raise to rollback the transaction
