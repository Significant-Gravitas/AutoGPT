import asyncio
import logging
from typing import Any, List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status
from prisma.enums import ReviewStatus

from backend.copilot.constants import legacy_chat_session_id, parse_node_id_from_exec_id
from backend.copilot.gate.chat_rules import set_answer_rules as set_chat_rules
from backend.copilot.gate.held import subject_keys as held_subject_keys
from backend.copilot.gate.held import wake as wake_for_held_calls
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    get_graph_execution_meta,
    get_node_executions,
)
from backend.data.graph import get_graph_settings
from backend.data.human_review import (
    create_auto_approval_record,
    get_pending_reviews_for_chat_session,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    get_reviews_by_node_exec_ids,
    has_pending_reviews_for_graph_exec,
    process_all_reviews_for_execution,
)
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.data.user import get_user_by_id
from backend.data.workspace import get_or_create_workspace
from backend.executor.utils import add_graph_execution

from .model import PendingHumanReviewModel, ReviewRequest, ReviewResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    tags=["v2", "executions", "review"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


async def _resolve_node_ids(
    node_exec_ids: list[str],
    graph_exec_id: str | None,
) -> dict[str, str]:
    """Resolve node_exec_id -> node_id for auto-approval records.

    A chat review's id encodes its node id as "{node_id}:{random}"; graph
    executions look up node_id from NodeExecution records.
    """
    if not node_exec_ids:
        return {}

    if graph_exec_id is None:
        return {neid: parse_node_id_from_exec_id(neid) for neid in node_exec_ids}

    node_execs = await get_node_executions(
        graph_exec_id=graph_exec_id, include_exec_data=False
    )
    node_exec_map = {ne.node_exec_id: ne.node_id for ne in node_execs}

    result = {}
    for neid in node_exec_ids:
        if neid in node_exec_map:
            result[neid] = node_exec_map[neid]
        else:
            logger.error(
                f"Failed to resolve node_id for {neid}: Node execution not found."
            )
    return result


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

    # Clients built before chat reviews had their own route ask for a chat
    # under its old synthetic id.
    if session_id := legacy_chat_session_id(graph_exec_id):
        return await get_pending_reviews_for_chat_session(session_id, user_id)

    graph_exec = await get_graph_execution_meta(
        user_id=user_id, execution_id=graph_exec_id
    )
    if not graph_exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph execution #{graph_exec_id} not found",
        )

    return await get_pending_reviews_for_execution(graph_exec_id, user_id)


@router.get(
    "/session/{chat_session_id}",
    summary="Get Pending Reviews for Chat Session",
    response_model=List[PendingHumanReviewModel],
    responses={
        200: {"description": "List of pending reviews for the chat session"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews_for_chat_session(
    chat_session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> List[PendingHumanReviewModel]:
    """Get the reviews an AutoPilot chat is waiting on, oldest first.

    Only the caller's own reviews are returned, so another user's chat session
    id yields an empty list.
    """
    return await get_pending_reviews_for_chat_session(chat_session_id, user_id)


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

    # Validate all reviews belong to the same execution or chat
    scopes = {
        (review.graph_exec_id, review.session_id) for review in reviews_map.values()
    }
    if len(scopes) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="All reviews in a single request must belong to the same execution.",
        )

    graph_exec_id, chat_session_id = next(iter(scopes))

    # Validate execution status for graph executions; a chat has none
    if graph_exec_id is not None:
        graph_exec_meta = await get_graph_execution_meta(
            user_id=user_id, execution_id=graph_exec_id
        )
        if not graph_exec_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph execution #{graph_exec_id} not found",
            )
        if graph_exec_meta.status not in (
            ExecutionStatus.REVIEW,
            ExecutionStatus.INCOMPLETE,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot process reviews while execution status is {graph_exec_meta.status}",
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

    # Read before the answer lands: a turn it wakes claims the held calls.
    chat_rule_keys = (
        await held_subject_keys(
            chat_session_id, [r.node_exec_id for r in request.reviews if r.chat_rule]
        )
        if chat_session_id is not None
        else {}
    )

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
                node_id=node_id,
                payload=review_result.payload,
                graph_exec_id=review_result.graph_exec_id,
                graph_id=review_result.graph_id,
                graph_version=review_result.graph_version,
                chat_session_id=review_result.session_id,
            )
            return (node_id, True)
        except Exception as e:
            logger.error(
                f"Failed to create auto-approval record for node {node_id}",
                exc_info=e,
            )
            return (node_id, False)

    # Collect node_exec_ids that need auto-approval and resolve their node_ids
    node_exec_ids_needing_auto_approval = [
        node_exec_id
        for node_exec_id, review_result in updated_reviews.items()
        if review_result.status == ReviewStatus.APPROVED
        and auto_approve_requests.get(node_exec_id, False)
    ]

    node_id_map = await _resolve_node_ids(
        node_exec_ids_needing_auto_approval, graph_exec_id
    )

    # Deduplicate by node_id — one auto-approval per node
    nodes_needing_auto_approval: dict[str, Any] = {}
    for node_exec_id in node_exec_ids_needing_auto_approval:
        node_id = node_id_map.get(node_exec_id)
        if node_id and node_id not in nodes_needing_auto_approval:
            nodes_needing_auto_approval[node_id] = updated_reviews[node_exec_id]

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
            auto_approval_failed_count += 1
            logger.error(
                f"Unexpected exception during auto-approval creation: {result}"
            )
        elif isinstance(result, tuple) and len(result) == 2 and not result[1]:
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

    if chat_session_id is not None and chat_rule_keys:
        await set_chat_rules(
            chat_session_id,
            updated_reviews,
            {review.node_exec_id: review.chat_rule for review in request.reviews},
            chat_rule_keys,
        )

    # A held call finishes on its own: the answer starts the chat's next turn.
    if chat_session_id is not None and updated_reviews:
        await wake_for_held_calls(user_id, chat_session_id, updated_reviews.values())

    # Resume graph execution only for real graph executions; a chat is resumed
    # by the LLM calling resume_capability with the review_id
    if graph_exec_id is not None and updated_reviews:
        still_has_pending = await has_pending_reviews_for_graph_exec(graph_exec_id)

        if not still_has_pending:
            first_review = next(iter(updated_reviews.values()))
            assert first_review.graph_id and first_review.graph_version is not None

            try:
                user = await get_user_by_id(user_id)
                settings = await get_graph_settings(
                    user_id=user_id,
                    graph_id=first_review.graph_id,
                    graph_version=first_review.graph_version,
                )

                user_timezone = (
                    user.timezone if user.timezone != USER_TIMEZONE_NOT_SET else "UTC"
                )

                workspace = await get_or_create_workspace(user_id)

                execution_context = ExecutionContext(
                    human_in_the_loop_safe_mode=settings.human_in_the_loop_safe_mode,
                    sensitive_action_safe_mode=settings.sensitive_action_safe_mode,
                    user_timezone=user_timezone,
                    workspace_id=workspace.id,
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
