"""
Data layer for Human In The Loop (HITL) review operations.
Handles all database operations for pending human reviews.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from prisma.enums import ReviewStatus
from prisma.models import AgentNodeExecution, PendingHumanReview
from prisma.types import PendingHumanReviewUpdateInput
from pydantic import BaseModel

from backend.api.features.executions.review.model import (
    PendingHumanReviewModel,
    SafeJsonData,
)
from backend.data.execution import get_graph_execution_meta
from backend.util.json import SafeJson

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    """Result of a review operation."""

    data: Optional[SafeJsonData] = None
    status: ReviewStatus
    message: str = ""
    processed: bool
    node_exec_id: str


def get_auto_approve_key(graph_exec_id: str, node_id: str) -> str:
    """Generate the special nodeExecId key for auto-approval records."""
    return f"auto_approve_{graph_exec_id}_{node_id}"


async def check_approval(
    node_exec_id: str,
    graph_exec_id: str,
    node_id: str,
    user_id: str,
    input_data: SafeJsonData | None = None,
) -> Optional[ReviewResult]:
    """
    Check if there's an existing approval for this node execution.

    Checks both:
    1. Normal approval by node_exec_id (previous run of the same node execution)
    2. Auto-approval by special key pattern "auto_approve_{graph_exec_id}_{node_id}"

    Args:
        node_exec_id: ID of the node execution
        graph_exec_id: ID of the graph execution
        node_id: ID of the node definition (not execution)
        user_id: ID of the user (for data isolation)
        input_data: Current input data (used for auto-approvals to avoid stale data)

    Returns:
        ReviewResult if approval found (either normal or auto), None otherwise
    """
    auto_approve_key = get_auto_approve_key(graph_exec_id, node_id)

    # Check for either normal approval or auto-approval in a single query
    existing_review = await PendingHumanReview.prisma().find_first(
        where={
            "OR": [
                {"nodeExecId": node_exec_id},
                {"nodeExecId": auto_approve_key},
            ],
            "status": ReviewStatus.APPROVED,
            "userId": user_id,
        },
    )

    if existing_review:
        is_auto_approval = existing_review.nodeExecId == auto_approve_key
        logger.info(
            f"Found {'auto-' if is_auto_approval else ''}approval for node {node_id} "
            f"(exec: {node_exec_id}) in execution {graph_exec_id}"
        )
        # For auto-approvals, use current input_data to avoid replaying stale payload
        # For normal approvals, use the stored payload (which may have been edited)
        return ReviewResult(
            data=(
                input_data
                if is_auto_approval and input_data is not None
                else existing_review.payload
            ),
            status=ReviewStatus.APPROVED,
            message=(
                "Auto-approved (user approved all future actions for this node)"
                if is_auto_approval
                else existing_review.reviewMessage or ""
            ),
            processed=True,
            node_exec_id=existing_review.nodeExecId,
        )

    return None


async def create_auto_approval_record(
    user_id: str,
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    node_id: str,
    payload: SafeJsonData,
) -> None:
    """
    Create an auto-approval record for a node in this execution.

    This is stored as a PendingHumanReview with a special nodeExecId pattern
    and status=APPROVED, so future executions of the same node can skip review.

    Raises:
        ValueError: If the graph execution doesn't belong to the user
    """
    # Validate that the graph execution belongs to this user (defense in depth)
    graph_exec = await get_graph_execution_meta(
        user_id=user_id, execution_id=graph_exec_id
    )
    if not graph_exec:
        raise ValueError(
            f"Graph execution {graph_exec_id} not found or doesn't belong to user {user_id}"
        )

    auto_approve_key = get_auto_approve_key(graph_exec_id, node_id)

    await PendingHumanReview.prisma().upsert(
        where={"nodeExecId": auto_approve_key},
        data={
            "create": {
                "nodeExecId": auto_approve_key,
                "userId": user_id,
                "graphExecId": graph_exec_id,
                "graphId": graph_id,
                "graphVersion": graph_version,
                "payload": SafeJson(payload),
                "instructions": "Auto-approval record",
                "editable": False,
                "status": ReviewStatus.APPROVED,
                "processed": True,
                "reviewedAt": datetime.now(timezone.utc),
            },
            "update": {},  # Already exists, no update needed
        },
    )


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


async def get_pending_review_by_node_exec_id(
    node_exec_id: str, user_id: str
) -> Optional["PendingHumanReviewModel"]:
    """
    Get a pending review by its node execution ID.

    Args:
        node_exec_id: The node execution ID to look up
        user_id: User ID for authorization (only returns if review belongs to this user)

    Returns:
        The pending review if found and belongs to user, None otherwise
    """
    review = await PendingHumanReview.prisma().find_first(
        where={
            "nodeExecId": node_exec_id,
            "userId": user_id,
            "status": ReviewStatus.WAITING,
        }
    )

    if not review:
        return None

    # Local import to avoid event loop conflicts in tests
    from backend.data.execution import get_node_execution

    node_exec = await get_node_execution(review.nodeExecId)
    node_id = node_exec.node_id if node_exec else review.nodeExecId
    return PendingHumanReviewModel.from_db(review, node_id=node_id)


async def get_reviews_by_node_exec_ids(
    node_exec_ids: list[str], user_id: str
) -> dict[str, "PendingHumanReviewModel"]:
    """
    Get multiple reviews by their node execution IDs regardless of status.

    Unlike get_pending_reviews_by_node_exec_ids, this returns reviews in any status
    (WAITING, APPROVED, REJECTED). Used for validation in idempotent operations.

    Args:
        node_exec_ids: List of node execution IDs to look up
        user_id: User ID for authorization (only returns reviews belonging to this user)

    Returns:
        Dictionary mapping node_exec_id -> PendingHumanReviewModel for found reviews
    """
    if not node_exec_ids:
        return {}

    reviews = await PendingHumanReview.prisma().find_many(
        where={
            "nodeExecId": {"in": node_exec_ids},
            "userId": user_id,
        }
    )

    if not reviews:
        return {}

    # Batch fetch all node executions to avoid N+1 queries
    node_exec_ids_to_fetch = [review.nodeExecId for review in reviews]
    node_execs = await AgentNodeExecution.prisma().find_many(
        where={"id": {"in": node_exec_ids_to_fetch}},
        include={"Node": True},
    )

    # Create mapping from node_exec_id to node_id
    node_exec_id_to_node_id = {
        node_exec.id: node_exec.agentNodeId for node_exec in node_execs
    }

    result = {}
    for review in reviews:
        node_id = node_exec_id_to_node_id.get(review.nodeExecId, review.nodeExecId)
        result[review.nodeExecId] = PendingHumanReviewModel.from_db(
            review, node_id=node_id
        )

    return result


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
        List of pending review models with node_id included
    """
    # Local import to avoid event loop conflicts in tests
    from backend.data.execution import get_node_execution

    # Calculate offset for pagination
    offset = (page - 1) * page_size

    reviews = await PendingHumanReview.prisma().find_many(
        where={"userId": user_id, "status": ReviewStatus.WAITING},
        order={"createdAt": "desc"},
        skip=offset,
        take=page_size,
    )

    # Fetch node_id for each review from NodeExecution
    result = []
    for review in reviews:
        node_exec = await get_node_execution(review.nodeExecId)
        node_id = node_exec.node_id if node_exec else review.nodeExecId
        result.append(PendingHumanReviewModel.from_db(review, node_id=node_id))

    return result


async def get_pending_reviews_for_execution(
    graph_exec_id: str, user_id: str
) -> list["PendingHumanReviewModel"]:
    """
    Get all pending reviews for a specific graph execution.

    Args:
        graph_exec_id: Graph execution ID
        user_id: User ID for security validation

    Returns:
        List of pending review models with node_id included
    """
    # Local import to avoid event loop conflicts in tests
    from backend.data.execution import get_node_execution

    reviews = await PendingHumanReview.prisma().find_many(
        where={
            "userId": user_id,
            "graphExecId": graph_exec_id,
            "status": ReviewStatus.WAITING,
        },
        order={"createdAt": "asc"},
    )

    # Fetch node_id for each review from NodeExecution
    result = []
    for review in reviews:
        node_exec = await get_node_execution(review.nodeExecId)
        node_id = node_exec.node_id if node_exec else review.nodeExecId
        result.append(PendingHumanReviewModel.from_db(review, node_id=node_id))

    return result


async def process_all_reviews_for_execution(
    user_id: str,
    review_decisions: dict[str, tuple[ReviewStatus, SafeJsonData | None, str | None]],
) -> dict[str, PendingHumanReviewModel]:
    """Process all pending reviews for an execution with approve/reject decisions.

    Handles race conditions gracefully: if a review was already processed with the
    same decision by a concurrent request, it's treated as success rather than error.

    Args:
        user_id: User ID for ownership validation
        review_decisions: Map of node_exec_id -> (status, reviewed_data, message)

    Returns:
        Dict of node_exec_id -> updated review model (includes already-processed reviews)
    """
    if not review_decisions:
        return {}

    node_exec_ids = list(review_decisions.keys())

    # Get all reviews (both WAITING and already processed) for the user
    all_reviews = await PendingHumanReview.prisma().find_many(
        where={
            "nodeExecId": {"in": node_exec_ids},
            "userId": user_id,
        },
    )

    # Separate into pending and already-processed reviews
    reviews_to_process = []
    already_processed = []
    for review in all_reviews:
        if review.status == ReviewStatus.WAITING:
            reviews_to_process.append(review)
        else:
            already_processed.append(review)

    # Check for truly missing reviews (not found at all)
    found_ids = {review.nodeExecId for review in all_reviews}
    missing_ids = set(node_exec_ids) - found_ids
    if missing_ids:
        raise ValueError(
            f"Reviews not found or access denied: {', '.join(missing_ids)}"
        )

    # Validate already-processed reviews have compatible status (same decision)
    # This handles race conditions where another request processed the same reviews
    for review in already_processed:
        requested_status = review_decisions[review.nodeExecId][0]
        if review.status != requested_status:
            raise ValueError(
                f"Review {review.nodeExecId} was already processed with status "
                f"{review.status}, cannot change to {requested_status}"
            )

    # Log if we're handling a race condition (some reviews already processed)
    if already_processed:
        already_processed_ids = [r.nodeExecId for r in already_processed]
        logger.info(
            f"Race condition handled: {len(already_processed)} review(s) already "
            f"processed by concurrent request: {already_processed_ids}"
        )

    # Create parallel update tasks for reviews that still need processing
    update_tasks = []

    for review in reviews_to_process:
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
    updated_reviews = await asyncio.gather(*update_tasks) if update_tasks else []

    # Note: Execution resumption is now handled at the API layer after ALL reviews
    # for an execution are processed (both approved and rejected)

    # Fetch node_id for each review and return as dict for easy access
    # Local import to avoid event loop conflicts in tests
    from backend.data.execution import get_node_execution

    # Combine updated reviews with already-processed ones (for idempotent response)
    all_result_reviews = list(updated_reviews) + already_processed

    result = {}
    for review in all_result_reviews:
        node_exec = await get_node_execution(review.nodeExecId)
        node_id = node_exec.node_id if node_exec else review.nodeExecId
        result[review.nodeExecId] = PendingHumanReviewModel.from_db(
            review, node_id=node_id
        )

    return result


async def update_review_processed_status(node_exec_id: str, processed: bool) -> None:
    """Update the processed status of a review."""
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": node_exec_id}, data={"processed": processed}
    )


async def cancel_pending_reviews_for_execution(graph_exec_id: str, user_id: str) -> int:
    """
    Cancel all pending reviews for a graph execution (e.g., when execution is stopped).

    Marks all WAITING reviews as REJECTED with a message indicating the execution was stopped.

    Args:
        graph_exec_id: The graph execution ID
        user_id: User ID who owns the execution (for security validation)

    Returns:
        Number of reviews cancelled

    Raises:
        ValueError: If the graph execution doesn't belong to the user
    """
    # Validate user ownership before cancelling reviews
    graph_exec = await get_graph_execution_meta(
        user_id=user_id, execution_id=graph_exec_id
    )
    if not graph_exec:
        raise ValueError(
            f"Graph execution {graph_exec_id} not found or doesn't belong to user {user_id}"
        )

    result = await PendingHumanReview.prisma().update_many(
        where={
            "graphExecId": graph_exec_id,
            "userId": user_id,
            "status": ReviewStatus.WAITING,
        },
        data={
            "status": ReviewStatus.REJECTED,
            "reviewMessage": "Execution was stopped by user",
            "processed": True,
            "reviewedAt": datetime.now(timezone.utc),
        },
    )
    return result
