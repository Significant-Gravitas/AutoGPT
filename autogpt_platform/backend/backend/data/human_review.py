"""
Data layer for Human In The Loop (HITL) review operations.
Handles all database operations for pending human reviews.
"""

from typing import Optional

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from pydantic import BaseModel

from backend.server.v2.executions.review.model import (
    PendingHumanReviewModel,
    SafeJsonData,
)
from backend.util.json import SafeJson


class ReviewResult(BaseModel):
    """Result of a review operation."""

    data: SafeJsonData
    status: str
    message: str = ""


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


async def get_or_upsert_human_review(
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
    Get existing completed review or upsert a pending review entry.

    This function either returns completed review results or creates/updates pending reviews.

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
    # Check if there's already a review for this node execution
    existing_review = await get_pending_review_by_node_exec_id(node_exec_id, user_id)

    if existing_review:
        if existing_review.status == "APPROVED":
            # Return the approved review result
            return ReviewResult(
                data=existing_review.payload,
                status="approved",
                message=existing_review.review_message or "",
            )
        elif existing_review.status == "REJECTED":
            # Return the rejected review result
            return ReviewResult(
                data=None,
                status="rejected",
                message=existing_review.review_message or "",
            )
        elif existing_review.status == "WAITING":
            # Review is already pending - don't overwrite to prevent race condition
            # User may be actively reviewing or editing the data in the UI
            return None

    # Create the pending review (only if no existing review)
    # With the new flat structure, we store payload, instructions, editable separately

    await PendingHumanReview.prisma().upsert(
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
            "update": {
                "payload": SafeJson(input_data),
                "instructions": message,
                "editable": editable,
                "status": ReviewStatus.WAITING,
            },
        },
    )

    # Return None to indicate we're waiting for human input
    return None


async def has_pending_review(node_exec_id: str) -> bool:
    """
    Check if a node execution has a pending review waiting for input.

    Args:
        node_exec_id: The node execution ID to check

    Returns:
        True if there is a review with WAITING status, False otherwise
    """
    review = await PendingHumanReview.prisma().find_first(
        where={"nodeExecId": node_exec_id, "status": ReviewStatus.WAITING}
    )
    return review is not None
