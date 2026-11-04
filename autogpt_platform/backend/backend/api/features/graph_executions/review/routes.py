import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Annotated, List

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Depends, HTTPException, Query, Security, status

from backend.api.live_auth import live_dependency
from backend.copilot.constants import legacy_chat_session_id
from backend.copilot.gate.held import answered_cards as held_answered_cards
from backend.copilot.gate.held import wake as wake_for_held_calls
from backend.copilot.model import ChatSessionInfo, get_chat_session_metadata
from backend.data.entitlement import enforce_payment_paywall
from backend.data.execution import (
    ExecutionStatus,
    get_graph_execution_exact_scope,
    get_graph_execution_meta,
)
from backend.data.human_review import (
    get_pending_reviews_for_chat_session,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    get_reviews_by_node_exec_ids,
)
from backend.data.tenancy import get_user_team_ids, live_resource_access_barrier
from backend.util.models import MAX_PAGE

from .model import PendingHumanReviewModel, ReviewRequest, ReviewResponse
from .service import process_reviews

logger = logging.getLogger(__name__)


router = APIRouter(
    tags=["v2", "executions", "review"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


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
    page: int = Query(1, ge=1, le=MAX_PAGE, description="Page number (1-indexed)"),
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

    # Clients built before chat reviews had their own route ask for a chat
    # under its old synthetic id.
    if session_id := legacy_chat_session_id(graph_exec_id):
        return await _list_pending_chat_reviews(session_id, user_id, ctx)

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

    async with live_resource_access_barrier(
        user_id, organization_id, team_id, "view"
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
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


@router.get(
    "/session/{chat_session_id}",
    summary="Get Pending Reviews for Chat Session",
    response_model=List[PendingHumanReviewModel],
    responses={
        200: {"description": "List of pending reviews for the chat session"},
        404: {"description": "Chat session not found in this workspace"},
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_pending_reviews_for_chat_session(
    chat_session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.get_request_context
    ),
) -> List[PendingHumanReviewModel]:
    """Get the reviews an AutoPilot chat is waiting on, oldest first.

    Only the caller's own reviews are returned, so another user's chat session
    id yields an empty list.
    """
    return await _list_pending_chat_reviews(chat_session_id, user_id, ctx)


@router.get(
    "/session/{chat_session_id}/answered",
    summary="Get Answered Reviews Waiting to Run for Chat Session",
    response_model=List[PendingHumanReviewModel],
)
async def list_answered_reviews_for_chat_session(
    chat_session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.get_request_context
    ),
) -> List[PendingHumanReviewModel]:
    """The chat's answered cards whose calls have not run yet, oldest first, so
    its box still shows them as answered, with Close and continue, after a reload."""
    session = await _chat_session_in_scope(chat_session_id, user_id, ctx)
    if session is None:
        return []
    async with live_resource_access_barrier(
        user_id, session.organization_id, session.team_id, "view"
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        return [
            row
            for row in await held_answered_cards(user_id, chat_session_id)
            if (row.organization_id, row.team_id)
            == (session.organization_id, session.team_id)
        ]


async def _list_pending_chat_reviews(
    chat_session_id: str,
    user_id: str,
    ctx: autogpt_auth_lib.RequestContext,
) -> List[PendingHumanReviewModel]:
    session = await _chat_session_in_scope(chat_session_id, user_id, ctx)
    if session is None:
        return []
    async with live_resource_access_barrier(
        user_id, session.organization_id, session.team_id, "view"
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        return await get_pending_reviews_for_chat_session(
            chat_session_id,
            user_id,
            organization_id=session.organization_id,
            team_id=session.team_id,
            enforce_scope=True,
        )


async def _chat_session_in_scope(
    chat_session_id: str,
    user_id: str,
    ctx: autogpt_auth_lib.RequestContext,
) -> ChatSessionInfo | None:
    """The caller's chat, or None when they have no chat by that id.

    A chat of theirs that lives in another organization or team than the
    request's is not found here, like any other resource outside the scope.
    """
    session = await get_chat_session_metadata(chat_session_id, user_id)
    if session is None:
        return None
    if session.organization_id != ctx.org_id or (
        ctx.team_id is not None and session.team_id != ctx.team_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session #{chat_session_id} not found",
        )
    return session


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
    is_copilot = chat_session_id is not None
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
            or (review.graph_exec_id, review.session_id)
            != (graph_exec_id, chat_session_id)
            for review in current.values()
        ):
            raise HTTPException(status_code=404, detail="Review not found")
        if not is_copilot:
            if graph_exec_id is None:
                # A review is anchored to a run or a chat; one with neither
                # cannot be decided here.
                raise HTTPException(status_code=404, detail="Review not found")
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


@router.post(
    "/action",
    response_model=ReviewResponse,
    dependencies=[Depends(enforce_payment_paywall)],
)
async def process_review_action(
    request: ReviewRequest,
    reviews_map: Annotated[
        dict[str, PendingHumanReviewModel],
        live_dependency(_live_review_action_dependency),
    ],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ReviewResponse:
    """Process reviews with approve or reject actions."""
    # The dependency holds every review to the one workspace it checked live.
    scope = next(iter(reviews_map.values()))
    outcome = await process_reviews(
        user_id,
        request.reviews,
        organization_id=scope.organization_id,
        team_id=scope.team_id,
        wait_for_other_cards=request.wait_for_other_cards,
    )

    error_message = None
    if outcome.auto_approval_failed_count:
        error_message = (
            f"{outcome.auto_approval_failed_count} auto-approval setting(s) could "
            f"not be saved. You may need to manually approve these reviews in "
            f"future executions."
        )

    return ReviewResponse(
        approved_count=outcome.approved_count,
        rejected_count=outcome.rejected_count,
        failed_count=outcome.auto_approval_failed_count,
        error=error_message,
        awaiting_other_cards=outcome.awaiting_other_cards,
    )


@router.post(
    "/session/{chat_session_id}/continue",
    summary="Continue Chat With Answered Reviews",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(enforce_payment_paywall)],
)
async def continue_with_answered_reviews(
    chat_session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.requires_resource_permission(
            autogpt_auth_lib.OrgAction.EXECUTE_RESOURCES,
            autogpt_auth_lib.TeamAction.EXECUTE_AGENTS,
        )
    ),
) -> None:
    """Start the chat's turn with the cards answered so far; the rest stay open
    and the turn is told how many still wait."""
    session = await _chat_session_in_scope(chat_session_id, user_id, ctx)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session #{chat_session_id} not found",
        )
    async with live_resource_access_barrier(
        user_id, session.organization_id, session.team_id, "execute"
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        await wake_for_held_calls(user_id, chat_session_id)
