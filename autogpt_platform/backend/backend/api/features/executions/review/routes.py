import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Annotated, Any, List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status
from prisma.enums import ReviewStatus

from backend.api.live_auth import live_dependency
from backend.copilot.constants import (
    is_copilot_synthetic_id,
    parse_node_id_from_exec_id,
)
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    get_graph_execution_exact_scope,
    get_graph_execution_meta,
    get_node_executions,
)
from backend.data.graph import get_graph_settings
from backend.data.human_review import (
    create_auto_approval_record,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    get_reviews_by_node_exec_ids,
    process_all_reviews_for_execution,
)
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.data.tenancy import get_user_team_ids, live_resource_access_barrier
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
    graph_exec_id: str,
    is_copilot: bool,
) -> dict[str, str]:
    """Resolve node_exec_id -> node_id for auto-approval records.

    CoPilot synthetic IDs encode node_id in the format "{node_id}:{random}".
    Graph executions look up node_id from NodeExecution records.
    """
    if not node_exec_ids:
        return {}

    if is_copilot:
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
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.requires_resource_permission(
            autogpt_auth_lib.OrgAction.VIEW_RESOURCES,
            autogpt_auth_lib.TeamAction.VIEW_AGENTS,
        )
    ),
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

    team_ids = (
        [ctx.team_id]
        if ctx.team_id is not None
        else await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []
    )
    scopes: list[tuple[str | None, str | None]] = [(ctx.org_id, None)]
    scopes.extend((ctx.org_id, team_id) for team_id in sorted(set(team_ids)))
    async with AsyncExitStack() as stack:
        for organization_id, team_id in scopes:
            allowed = await stack.enter_async_context(
                live_resource_access_barrier(
                    user_id,
                    organization_id,
                    team_id,
                    "view",
                )
            )
            if not allowed:
                raise HTTPException(status_code=403, detail="Resource access revoked")

        return await get_pending_reviews_for_user(
            user_id,
            page,
            page_size,
            organization_id=ctx.org_id,
            team_ids=team_ids,
            team_id_restriction=ctx.team_id,
        )


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
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.requires_resource_permission(
            autogpt_auth_lib.OrgAction.VIEW_RESOURCES,
            autogpt_auth_lib.TeamAction.VIEW_AGENTS,
        )
    ),
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

    # Verify user owns the graph execution before returning reviews
    # (CoPilot synthetic IDs don't have graph execution records)
    if not is_copilot_synthetic_id(graph_exec_id):
        candidate = await get_graph_execution_meta(
            user_id=user_id,
            execution_id=graph_exec_id,
            organization_id=ctx.org_id,
            team_id_restriction=ctx.team_id,
        )
        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph execution #{graph_exec_id} not found",
            )
        organization_id, team_id = candidate.organization_id, candidate.team_id
    else:
        candidates = await get_pending_reviews_for_execution(graph_exec_id, user_id)
        scopes = {(review.organization_id, review.team_id) for review in candidates}
        if len(scopes) != 1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph execution #{graph_exec_id} not found",
            )
        organization_id, team_id = next(iter(scopes))
        if organization_id != ctx.org_id or (
            ctx.team_id is not None and team_id != ctx.team_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph execution #{graph_exec_id} not found",
            )

    async with live_resource_access_barrier(
        user_id, organization_id, team_id, "view"
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        if not is_copilot_synthetic_id(graph_exec_id):
            current = await get_graph_execution_exact_scope(
                user_id,
                graph_exec_id,
                organization_id,
                team_id,
            )
            if current is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Graph execution #{graph_exec_id} not found",
                )
        return await get_pending_reviews_for_execution(
            graph_exec_id,
            user_id,
            organization_id=organization_id,
            team_id=team_id,
            enforce_scope=True,
        )


async def _live_review_action_dependency(
    request: ReviewRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.requires_resource_permission(
            autogpt_auth_lib.OrgAction.EXECUTE_RESOURCES,
            autogpt_auth_lib.TeamAction.EXECUTE_AGENTS,
        )
    ),
) -> AsyncIterator[dict[str, PendingHumanReviewModel]]:
    all_request_node_ids = {review.node_exec_id for review in request.reviews}
    if not all_request_node_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one review must be provided",
        )
    reviews_map = await get_reviews_by_node_exec_ids(
        list(all_request_node_ids), user_id
    )
    missing_ids = all_request_node_ids - set(reviews_map.keys())
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review(s) not found: {', '.join(missing_ids)}",
        )
    graph_exec_ids = {review.graph_exec_id for review in reviews_map.values()}
    if len(graph_exec_ids) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="All reviews in a single request must belong to the same execution.",
        )
    graph_exec_id = next(iter(graph_exec_ids))
    is_copilot = is_copilot_synthetic_id(graph_exec_id)
    scopes = {
        (review.organization_id, review.team_id) for review in reviews_map.values()
    }
    if len(scopes) != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="All reviews in a single request must belong to the same workspace.",
        )
    organization_id, team_id = next(iter(scopes))
    if organization_id != ctx.org_id or (
        ctx.team_id is not None and team_id != ctx.team_id
    ):
        raise HTTPException(status_code=404, detail="Review not found")

    async with live_resource_access_barrier(
        user_id, organization_id, team_id, "execute"
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        current = await get_reviews_by_node_exec_ids(
            list(all_request_node_ids), user_id
        )
        if set(current) != all_request_node_ids or any(
            (review.organization_id, review.team_id) != (organization_id, team_id)
            or review.graph_exec_id != graph_exec_id
            for review in current.values()
        ):
            raise HTTPException(status_code=404, detail="Review not found")
        if not is_copilot:
            graph_exec_meta = await get_graph_execution_exact_scope(
                user_id,
                graph_exec_id,
                organization_id,
                team_id,
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
                    detail=(
                        "Cannot process reviews while execution status is "
                        f"{graph_exec_meta.status}"
                    ),
                )
        yield current


@router.post("/action", response_model=ReviewResponse)
async def process_review_action(
    request: ReviewRequest,
    reviews_map: Annotated[
        dict[str, PendingHumanReviewModel],
        live_dependency(_live_review_action_dependency),
    ],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ReviewResponse:
    """Process reviews with approve or reject actions."""

    graph_exec_ids = {review.graph_exec_id for review in reviews_map.values()}
    graph_exec_id = next(iter(graph_exec_ids))
    is_copilot = is_copilot_synthetic_id(graph_exec_id)

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
                graph_exec_id=review_result.graph_exec_id,
                graph_id=review_result.graph_id,
                graph_version=review_result.graph_version,
                node_id=node_id,
                payload=review_result.payload,
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
        node_exec_ids_needing_auto_approval, graph_exec_id, is_copilot
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

    # Resume graph execution only for real graph executions (not CoPilot)
    # CoPilot sessions are resumed by the LLM retrying run_block with review_id
    if not is_copilot and updated_reviews.should_resume:
        first_review = next(iter(updated_reviews.values()))

        try:
            user = await get_user_by_id(user_id)
            settings = await get_graph_settings(
                user_id=user_id, graph_id=first_review.graph_id
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
                organization_id=first_review.organization_id,
                team_id=first_review.team_id,
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
