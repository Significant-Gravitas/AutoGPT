"""
Data layer for Human In The Loop (HITL) review operations.
Handles all database operations for pending human reviews.
"""

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from pydantic import BaseModel

from backend.data.execution import NodeExecutionResult
from backend.server.v2.executions.review.model import (
    PendingHumanReviewModel,
    SafeJsonData,
)
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    """Result of a review operation."""

    data: SafeJsonData
    status: ReviewStatus
    message: str = ""
    processed: bool
    node_exec_id: str


async def get_pending_review_by_node_exec_id(
    node_exec_id: str, user_id: str
) -> Optional["PendingHumanReviewModel"]:
    """
    Get a pending review by node execution ID with user ownership validation.

    Args:
        node_exec_id: The node execution ID to check
        user_id: The user ID to validate ownership

    Returns:
        The existing review if found and owned by the user, None otherwise
    """
    review = await PendingHumanReview.prisma().find_first(
        where={
            "nodeExecId": node_exec_id,
            "userId": user_id,
        }
    )

    if review:
        return PendingHumanReviewModel.from_db(review)

    return None


async def get_or_create_human_review(
    user_id: str,
    node_exec_id: str,
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    input_data: SafeJsonData,
    message: str,
    editable: bool,
) -> Optional[ReviewResult]:
    """
    Get existing review or create a new pending review entry.

    Uses upsert with empty update to get existing or create new review in a single operation.

    Args:
        user_id: ID of the user who owns this review
        node_exec_id: ID of the node execution
        graph_exec_id: ID of the graph execution
        graph_id: ID of the graph template
        graph_version: Version of the graph template
        input_data: The data to be reviewed
        message: Instructions for the reviewer
        editable: Whether the data can be edited

    Returns:
        ReviewResult if the review is complete, None if waiting for human input
    """
    try:
        logger.debug(f"Getting or creating review for node {node_exec_id}")

        # Upsert - get existing or create new review
        review = await PendingHumanReview.prisma().upsert(
            where={"nodeExecId": node_exec_id},
            data={
                "create": {
                    "userId": user_id,
                    "nodeExecId": node_exec_id,
                    "graphExecId": graph_exec_id,
                    "graphId": graph_id,
                    "graphVersion": graph_version,
                    "payload": SafeJson(input_data),
                    "instructions": message,
                    "editable": editable,
                    "status": ReviewStatus.WAITING,
                },
                "update": {},  # Do nothing on update - keep existing review as is
            },
        )

        logger.info(
            f"Review {'created' if review.createdAt == review.updatedAt else 'retrieved'} for node {node_exec_id} with status {review.status}"
        )
    except Exception as e:
        logger.error(
            f"Database error in get_or_create_human_review for node {node_exec_id}: {str(e)}"
        )
        raise

    # Early return if already processed
    if review.processed:
        return None

    if review.status == ReviewStatus.APPROVED:
        # Return the approved review result
        return ReviewResult(
            data=review.payload,
            status=ReviewStatus.APPROVED,
            message=review.reviewMessage or "",
            processed=review.processed,
            node_exec_id=review.nodeExecId,
        )
    elif review.status == ReviewStatus.REJECTED:
        # Return the rejected review result
        return ReviewResult(
            data=None,
            status=ReviewStatus.REJECTED,
            message=review.reviewMessage or "",
            processed=review.processed,
            node_exec_id=review.nodeExecId,
        )
    else:
        # Review is pending - return None to continue waiting
        return None


async def has_pending_reviews_for_graph_exec(graph_exec_id: str) -> bool:
    """
    Check if a graph execution has any pending reviews.

    Args:
        graph_exec_id: The graph execution ID to check

    Returns:
        True if there are reviews waiting for human input, False otherwise
    """
    # Check if there are any reviews waiting for human input
    count = await PendingHumanReview.prisma().count(
        where={"graphExecId": graph_exec_id, "status": ReviewStatus.WAITING}
    )
    return count > 0


async def get_unprocessed_review_node_executions(
    graph_exec_id: str,
) -> list[NodeExecutionResult]:
    """
    Get node executions for nodes with unprocessed completed reviews.

    Args:
        graph_exec_id: The graph execution ID to check

    Returns:
        List of NodeExecutionResult for ready-to-execute nodes
    """
    # Get all unprocessed reviews with their node executions in a single query
    reviews = await PendingHumanReview.prisma().find_many(
        where={
            "graphExecId": graph_exec_id,
            "OR": [
                {"status": ReviewStatus.APPROVED},
                {"status": ReviewStatus.REJECTED},
            ],
            "processed": False,
        },
        include={"NodeExecution": True},
    )

    if not reviews:
        return []

    # Convert to NodeExecutionResult
    node_executions = []
    for review in reviews:
        if review.NodeExecution:
            node_exec = NodeExecutionResult.from_db(review.NodeExecution)
            node_executions.append(node_exec)

    return node_executions


async def get_pending_reviews_for_user(
    user_id: str, page: int = 1, page_size: int = 25
) -> list["PendingHumanReviewModel"]:
    """
    Get all pending reviews for a user with pagination.

    Args:
        user_id: User ID to get reviews for
        page: Page number (1-indexed)
        page_size: Number of reviews per page

    Returns:
        List of pending review models
    """
    # Calculate offset for pagination
    offset = (page - 1) * page_size

    reviews = await PendingHumanReview.prisma().find_many(
        where={"userId": user_id, "status": ReviewStatus.WAITING},
        order={"createdAt": "desc"},
        skip=offset,
        take=page_size,
    )

    return [PendingHumanReviewModel.from_db(review) for review in reviews]


async def get_pending_reviews_for_execution(
    graph_exec_id: str, user_id: str
) -> list["PendingHumanReviewModel"]:
    """
    Get all pending reviews for a specific graph execution.

    Args:
        graph_exec_id: Graph execution ID
        user_id: User ID for security validation

    Returns:
        List of pending review models
    """
    reviews = await PendingHumanReview.prisma().find_many(
        where={
            "userId": user_id,
            "graphExecId": graph_exec_id,
            "status": ReviewStatus.WAITING,
        },
        order={"createdAt": "asc"},
    )

    return [PendingHumanReviewModel.from_db(review) for review in reviews]


async def update_review_action(
    node_exec_id: str,
    user_id: str,
    action: Literal["approve", "reject"],
    reviewed_data: SafeJsonData | None = None,
    message: str | None = None,
) -> Optional["PendingHumanReviewModel"]:
    """Update a review with approve/reject action."""

    # Input validation
    if not node_exec_id or not node_exec_id.strip():
        raise ValueError("node_exec_id cannot be empty")

    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")

    if action not in ["approve", "reject"]:
        raise ValueError("action must be 'approve' or 'reject'")

    # Validate message length
    if message and len(message) > 10_000:
        raise ValueError("Review message cannot exceed 10,000 characters")

    # Validate reviewed_data size
    if reviewed_data is not None:
        try:
            data_size = len(str(reviewed_data))
            if data_size > 1_000_000:  # 1MB limit
                raise ValueError("Reviewed data cannot exceed 1MB in size")
        except Exception:
            raise ValueError("Invalid reviewed data format")

    # Determine status from action
    new_status = ReviewStatus.APPROVED if action == "approve" else ReviewStatus.REJECTED

    try:
        logger.debug(f"Updating review for node {node_exec_id} with action {action}")

        # First check if review exists and is valid for update
        existing_review = await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": node_exec_id}
        )
    except Exception as e:
        logger.error(f"Database error while fetching review {node_exec_id}: {str(e)}")
        raise

    if not existing_review:
        logger.warning(f"Review not found for node {node_exec_id}")
        return None

    if existing_review.userId != user_id:
        logger.warning(
            f"Access denied: user {user_id} attempted to update review owned by {existing_review.userId}"
        )
        return None

    if existing_review.status != ReviewStatus.WAITING:
        logger.warning(
            f"Review {node_exec_id} is not in WAITING status (current: {existing_review.status})"
        )
        return None

    try:
        # Update the review
        if reviewed_data is not None:
            logger.info(
                f"Updating review {node_exec_id} with edited data (action: {action})"
            )
            updated_review = await PendingHumanReview.prisma().update(
                where={"nodeExecId": node_exec_id},
                data={
                    "status": new_status,
                    "payload": SafeJson(reviewed_data),
                    "reviewMessage": message,
                    "wasEdited": True,
                    "reviewedAt": datetime.now(timezone.utc),
                },
            )
        else:
            logger.info(
                f"Updating review {node_exec_id} without data changes (action: {action})"
            )
            updated_review = await PendingHumanReview.prisma().update(
                where={"nodeExecId": node_exec_id},
                data={
                    "status": new_status,
                    "reviewMessage": message,
                    "wasEdited": False,
                    "reviewedAt": datetime.now(timezone.utc),
                },
            )

        logger.info(
            f"Successfully updated review {node_exec_id} to status {new_status}"
        )
        if updated_review is None:
            raise ValueError(f"Failed to update review {node_exec_id}")
        return PendingHumanReviewModel.from_db(updated_review)

    except Exception as e:
        logger.error(f"Database error while updating review {node_exec_id}: {str(e)}")
        raise


async def update_review_processed_status(node_exec_id: str, processed: bool) -> None:
    """Update the processed status of a review."""
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": node_exec_id}, data={"processed": processed}
    )
