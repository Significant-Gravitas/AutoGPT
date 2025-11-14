"""
Data layer for Human In The Loop (HITL) review operations.
Handles all database operations for pending human reviews.
"""

from typing import Any, Optional

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from pydantic import BaseModel

from backend.util.json import SafeJson


async def get_pending_review_by_node(
    node_exec_id: str, user_id: str
) -> Optional[PendingHumanReview]:
    """
    Get a pending review by node execution ID with user ownership validation.

    Args:
        node_exec_id: The node execution ID to check
        user_id: The user ID to validate ownership

    Returns:
        The existing review if found and owned by the user, None otherwise
    """
    review = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": node_exec_id}
    )
    # Validate ownership for defense-in-depth security
    if review and review.userId != user_id:
        return None
    return review


async def upsert_pending_review(
    node_exec_id: str,
    user_id: str,
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    review_data: Any,
) -> None:
    """
    Create or update a pending review.

    Args:
        node_exec_id: ID of the node execution
        user_id: ID of the user who owns this review
        graph_exec_id: ID of the graph execution
        graph_id: ID of the graph template
        graph_version: Version of the graph template
        review_data: The structured review data to store
    """
    await PendingHumanReview.prisma().upsert(
        where={"nodeExecId": node_exec_id},
        data={
            "create": {
                "userId": user_id,
                "nodeExecId": node_exec_id,
                "graphExecId": graph_exec_id,
                "graphId": graph_id,
                "graphVersion": graph_version,
                "data": SafeJson(review_data),
                "status": ReviewStatus.WAITING,
            },
            "update": {
                "data": SafeJson(review_data),
                "status": ReviewStatus.WAITING,
            },
        },
    )


class ReviewDataStructure(BaseModel):
    """Structured representation of review data stored in the database."""

    data: Any
    message: str
    editable: bool


class ReviewResult(BaseModel):
    """Result of a review operation."""

    data: Any
    status: str
    message: str = ""


class HITLValidationError(Exception):
    """Exception raised when HITL validation fails."""

    def __init__(self, message: str, review_message: str = ""):
        super().__init__(message)
        self.review_message = review_message


def extract_approved_data(review: PendingHumanReview) -> Any:
    """
    Extract approved data from a review record.

    Handles both structured and legacy data formats.

    Args:
        review: The approved review record

    Returns:
        The extracted data
    """
    try:
        # Try to parse as structured format
        review_structure = ReviewDataStructure.model_validate(review.data)
        return review_structure.data
    except Exception:
        # Fallback for legacy data format or corrupted data
        if isinstance(review.data, dict) and "data" in review.data:
            return review.data["data"]
        else:
            return review.data


async def process_approved_review(
    review: PendingHumanReview, expected_data_type: type
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
    from backend.util.type import convert

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

    # Clean up the review record only after successful conversion
    await PendingHumanReview.prisma().delete(where={"id": review.id})

    return ReviewResult(
        data=approved_data, status="approved", message=review.reviewMessage or ""
    )


async def process_rejected_review(review: PendingHumanReview) -> ReviewResult:
    """
    Process a rejected review and clean up the database.

    Args:
        review: The rejected review to process

    Returns:
        ReviewResult with rejection details
    """
    # Clean up the review record
    await PendingHumanReview.prisma().delete(where={"id": review.id})

    return ReviewResult(
        data=None, status="rejected", message=review.reviewMessage or ""
    )


async def get_or_upsert_human_review(
    user_id: str,
    node_exec_id: str,
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    input_data: Any,
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
    existing_review = await get_pending_review_by_node(node_exec_id, user_id)

    if existing_review:
        if existing_review.status == ReviewStatus.APPROVED:
            # Process the approved review
            return await process_approved_review(existing_review, expected_data_type)
        elif existing_review.status == ReviewStatus.REJECTED:
            # Process the rejected review
            return await process_rejected_review(existing_review)
        elif existing_review.status == ReviewStatus.WAITING:
            # Review is already pending - don't overwrite to prevent race condition
            # User may be actively reviewing or editing the data in the UI
            return None

    # Create the pending review (only if no existing review)
    review_data = {
        "data": input_data,
        "message": message,
        "editable": editable,
    }

    await upsert_pending_review(
        node_exec_id=node_exec_id,
        user_id=user_id,
        graph_exec_id=graph_exec_id,
        graph_id=graph_id,
        graph_version=graph_version,
        review_data=review_data,
    )

    # Return None to indicate we're waiting for human input
    return None


async def has_pending_review(node_exec_id: str) -> bool:
    """
    Check if a node execution has a pending review.

    Args:
        node_exec_id: The node execution ID to check

    Returns:
        True if there is a pending review waiting, False otherwise
    """
    review = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": node_exec_id}
    )
    return review is not None and review.status == ReviewStatus.WAITING
