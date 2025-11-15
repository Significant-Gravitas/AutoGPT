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
from backend.util.type import convert


class HITLValidationError(ValueError):
    """Exception raised when HITL validation fails."""

    def __init__(self, message: str, review_message: str = ""):
        super().__init__(message)
        self.review_message = review_message


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


def extract_approved_data(review: PendingHumanReviewModel) -> SafeJsonData:
    """
    Extract approved data from a review record.

    Args:
        review: The approved review record

    Returns:
        The payload data from the review
    """
    # With the new flat structure, just return the payload directly
    return review.payload


async def process_approved_review(
    review: PendingHumanReviewModel, expected_data_type: type
) -> ReviewResult:
    """
    Process an approved review and clean up the database.

    Args:
        review: The approved review to process
        expected_data_type: The expected type for the data

    Returns:
        ReviewResult with the processed data

    Raises:
        HITLValidationError: If data conversion fails after approval
    """
    approved_data = extract_approved_data(review)

    try:
        approved_data = convert(approved_data, expected_data_type)
    except Exception as e:
        # Reset review back to WAITING status so user can fix the data
        await PendingHumanReview.prisma().update(
            where={"id": review.id}, data={"status": ReviewStatus.WAITING}
        )
        raise HITLValidationError(
            f"Failed to convert approved data to {expected_data_type.__name__}: {e}",
            review_message="Data conversion failed after approval. Please review and fix the data format.",
        )

    # Review completed successfully - status already updated to APPROVED

    return ReviewResult(
        data=approved_data, status="approved", message=review.review_message or ""
    )


async def process_rejected_review(review: PendingHumanReviewModel) -> ReviewResult:
    """
    Process a rejected review and clean up the database.

    Args:
        review: The rejected review to process

    Returns:
        ReviewResult with rejection details
    """
    # Review completed - status already updated to REJECTED

    return ReviewResult(
        data=None, status="rejected", message=review.review_message or ""
    )


async def get_or_upsert_human_review(
    user_id: str,
    node_exec_id: str,
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    input_data: SafeJsonData,
    message: str,
    editable: bool,
    expected_data_type: type,
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
        expected_data_type: Expected type for the output data

    Returns:
        ReviewResult if the review is complete, None if waiting for human input
    """
    # Check if there's already a review for this node execution
    existing_review = await get_pending_review_by_node_exec_id(node_exec_id, user_id)

    if existing_review:
        if existing_review.status == "APPROVED":
            # Process the approved review
            return await process_approved_review(existing_review, expected_data_type)
        elif existing_review.status == "REJECTED":
            # Process the rejected review
            return await process_rejected_review(existing_review)
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
                "data": SafeJson(input_data),
                "instructions": message,
                "editable": editable,
                "status": ReviewStatus.WAITING,
            },
            "update": {
                "data": SafeJson(input_data),
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
