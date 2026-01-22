"""
Data layer for Human In The Loop (HITL) review operations.
Handles all database operations for pending human reviews.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from prisma.types import PendingHumanReviewUpdateInput
from pydantic import BaseModel

from backend.api.features.executions.review.model import (
    PendingHumanReviewModel,
    SafeJsonData,
)
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    """Result of a review operation."""

    data: Optional[SafeJsonData] = None
    status: ReviewStatus
    message: str = ""
    processed: bool
    node_exec_id: str


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

    # If pending, return None to continue waiting, otherwise return the review result
    if review.status == ReviewStatus.WAITING:
        return None
    else:
        return ReviewResult(
            data=review.payload,
            status=review.status,
            message=review.reviewMessage or "",
            processed=review.processed,
            node_exec_id=review.nodeExecId,
        )


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


async def process_all_reviews_for_execution(
    user_id: str,
    review_decisions: dict[str, tuple[ReviewStatus, SafeJsonData | None, str | None]],
) -> dict[str, PendingHumanReviewModel]:
    """Process all pending reviews for an execution with approve/reject decisions.

    Args:
        user_id: User ID for ownership validation
        review_decisions: Map of node_exec_id -> (status, reviewed_data, message)

    Returns:
        Dict of node_exec_id -> updated review model
    """
    if not review_decisions:
        return {}

    node_exec_ids = list(review_decisions.keys())

    # Get all reviews for validation
    reviews = await PendingHumanReview.prisma().find_many(
        where={
            "nodeExecId": {"in": node_exec_ids},
            "userId": user_id,
            "status": ReviewStatus.WAITING,
        },
    )

    # Validate all reviews can be processed
    if len(reviews) != len(node_exec_ids):
        missing_ids = set(node_exec_ids) - {review.nodeExecId for review in reviews}
        raise ValueError(
            f"Reviews not found, access denied, or not in WAITING status: {', '.join(missing_ids)}"
        )

    # Create parallel update tasks
    update_tasks = []

    for review in reviews:
        new_status, reviewed_data, message = review_decisions[review.nodeExecId]
        has_data_changes = reviewed_data is not None and reviewed_data != review.payload

        # Check edit permissions for actual data modifications
        if has_data_changes and not review.editable:
            raise ValueError(f"Review {review.nodeExecId} is not editable")

        update_data: PendingHumanReviewUpdateInput = {
            "status": new_status,
            "reviewMessage": message,
            "wasEdited": has_data_changes,
            "reviewedAt": datetime.now(timezone.utc),
        }

        if has_data_changes:
            update_data["payload"] = SafeJson(reviewed_data)

        task = PendingHumanReview.prisma().update(
            where={"nodeExecId": review.nodeExecId},
            data=update_data,
        )
        update_tasks.append(task)

    # Execute all updates in parallel and get updated reviews
    updated_reviews = await asyncio.gather(*update_tasks)

    # Note: Execution resumption is now handled at the API layer after ALL reviews
    # for an execution are processed (both approved and rejected)

    # Return as dict for easy access
    return {
        review.nodeExecId: PendingHumanReviewModel.from_db(review)
        for review in updated_reviews
    }


async def update_review_processed_status(node_exec_id: str, processed: bool) -> None:
    """Update the processed status of a review."""
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": node_exec_id}, data={"processed": processed}
    )
