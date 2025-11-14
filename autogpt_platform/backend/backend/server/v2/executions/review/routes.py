import logging
from datetime import datetime, timezone
from typing import Any, List, Literal, cast

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Security, status

from backend.server.v2.executions.review.model import (
    PendingHumanReviewResponse,
    ReviewActionRequest,
)

logger = logging.getLogger(__name__)

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
    """Get all pending reviews for the current user."""
    from prisma.models import PendingHumanReview

    reviews = await PendingHumanReview.prisma().find_many(
        where={"userId": user_id, "status": "WAITING"},
        order={"createdAt": "desc"},
    )

    return [
        PendingHumanReviewResponse(
            id=review.id,
            user_id=review.userId,
            node_exec_id=review.nodeExecId,
            graph_exec_id=review.graphExecId,
            graph_id=review.graphId,
            graph_version=review.graphVersion,
            data=review.data,
            status=cast(Literal["WAITING", "APPROVED", "REJECTED"], review.status),
            review_message=review.reviewMessage,
            created_at=review.createdAt,
            updated_at=review.updatedAt,
            reviewed_at=review.reviewedAt,
        )
        for review in reviews
    ]


@router.get(
    "/execution/{graph_exec_id}",
    summary="Get Pending Reviews for Execution",
    responses={
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews_for_execution(
    graph_exec_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> List[PendingHumanReviewResponse]:
    """Get all pending reviews for a specific graph execution."""
    from prisma.models import PendingHumanReview

    reviews = await PendingHumanReview.prisma().find_many(
        where={"userId": user_id, "graphExecId": graph_exec_id, "status": "WAITING"},
        order={"createdAt": "asc"},
    )

    return [
        PendingHumanReviewResponse(
            id=review.id,
            user_id=review.userId,
            node_exec_id=review.nodeExecId,
            graph_exec_id=review.graphExecId,
            graph_id=review.graphId,
            graph_version=review.graphVersion,
            data=review.data,
            status=cast(Literal["WAITING", "APPROVED", "REJECTED"], review.status),
            review_message=review.reviewMessage,
            created_at=review.createdAt,
            updated_at=review.updatedAt,
            reviewed_at=review.reviewedAt,
        )
        for review in reviews
    ]


@router.post(
    "/{review_id}/action",
    summary="Review Data",
    responses={
        404: {"description": "Review not found"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def review_data(
    review_id: str,
    request: ReviewActionRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> dict[str, Any]:
    """Approve or reject pending review data."""
    from prisma.models import PendingHumanReview

    # Find the review and verify ownership
    review = await PendingHumanReview.prisma().find_unique(where={"id": review_id})

    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Review not found"
        )

    if review.userId != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    if review.status != "WAITING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Review is already {review.status.lower()}",
        )

    # Update the review
    now = datetime.now(timezone.utc)
    update_status = "APPROVED" if request.action == "approve" else "REJECTED"

    # Handle reviewed_data for approve action
    review_data = review.data
    if request.action == "approve" and request.reviewed_data is not None:
        # Update only the data part while preserving the structure
        if isinstance(review.data, dict) and "data" in review.data:
            review_data = {
                **review.data,
                "data": request.reviewed_data,  # Update just the data field
            }
        else:
            # Fallback: replace entire data
            review_data = request.reviewed_data

    # Simple update without complex type handling
    from backend.util.json import SafeJson

    await PendingHumanReview.prisma().update(
        where={"id": review_id},
        data={
            "status": update_status,
            "data": SafeJson(review_data),  # Store the (possibly modified) data
            "reviewMessage": request.message,
            "reviewedAt": now,
        },
    )

    # If approved, trigger graph execution resume
    if request.action == "approve":
        await _resume_graph_execution(review.graphExecId)

    return {"status": "success", "action": request.action}


async def _resume_graph_execution(graph_exec_id: str) -> None:
    """Resume a graph execution by updating its status."""
    try:
        from backend.data.execution import ExecutionStatus, update_graph_execution_stats
        from backend.util.clients import get_database_manager_async_client

        # Get the graph execution details
        db = get_database_manager_async_client()
        graph_exec = await db.get_graph_execution_meta(
            user_id="", execution_id=graph_exec_id  # We'll validate user_id separately
        )

        if not graph_exec:
            logger.error(f"Graph execution {graph_exec_id} not found")
            return

        # Update the graph execution status to QUEUED so the scheduler picks it up
        await update_graph_execution_stats(
            graph_exec_id=graph_exec_id, status=ExecutionStatus.QUEUED
        )

        logger.info(f"Resumed graph execution {graph_exec_id}")

    except Exception as e:
        logger.error(f"Failed to resume graph execution {graph_exec_id}: {e}")
