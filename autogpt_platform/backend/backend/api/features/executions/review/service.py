"""Human-in-the-loop review processing, shared by the internal and v2 APIs.

One implementation so the two surfaces cannot disagree about which reviews a
request may touch, what a run's status has to be, or when execution resumes.
"""

import asyncio
import logging
from typing import Any, Optional, Sequence

from fastapi import HTTPException, status
from prisma.enums import ReviewStatus
from pydantic import BaseModel, Field

from backend.copilot.constants import (
    is_copilot_synthetic_id,
    parse_node_id_from_exec_id,
)
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    get_graph_execution_meta,
    get_node_executions,
)
from backend.data.graph import get_graph_settings
from backend.data.human_review import (
    create_auto_approval_record,
    get_reviews_by_node_exec_ids,
    has_pending_reviews_for_graph_exec,
    process_all_reviews_for_execution,
)
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.data.user import get_user_by_id
from backend.data.workspace import get_or_create_workspace
from backend.executor.utils import add_graph_execution

from .model import ReviewItem

logger = logging.getLogger(__name__)


class ReviewOutcome(BaseModel):
    """What processing a batch of review decisions did."""

    graph_exec_id: str
    approved_count: int
    rejected_count: int
    auto_approval_failed_count: int = Field(
        description="Approved reviews whose auto-approve-future record failed to save"
    )


async def process_reviews(
    user_id: str,
    reviews: Sequence[ReviewItem],
    *,
    graph_exec_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> ReviewOutcome:
    """Approve or reject reviews, then resume the run once none are pending.

    `graph_exec_id` pins the run the decisions must belong to, for a caller
    that already knows it from the URL; without it the run is taken from the
    reviews themselves, which must then all belong to one.
    """
    if not reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one review must be provided",
        )

    reviews_map = await get_reviews_by_node_exec_ids(
        [review.node_exec_id for review in reviews], user_id
    )
    missing_ids = {r.node_exec_id for r in reviews} - set(reviews_map)
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review(s) not found: {', '.join(sorted(missing_ids))}",
        )

    graph_exec_id = _one_execution(reviews_map.values(), graph_exec_id)
    is_copilot = is_copilot_synthetic_id(graph_exec_id)
    if not is_copilot:
        await _assert_awaiting_review(user_id, graph_exec_id)

    # An auto-approved review takes the original data: approving future runs of
    # a block is not the place to also edit this one's payload.
    updated_reviews = await process_all_reviews_for_execution(
        user_id=user_id,
        review_decisions={
            review.node_exec_id: (
                ReviewStatus.APPROVED if review.approved else ReviewStatus.REJECTED,
                None if review.auto_approve_future else review.reviewed_data,
                review.message,
            )
            for review in reviews
        },
    )

    failed = await _record_auto_approvals(
        user_id,
        {r.node_exec_id: r.auto_approve_future for r in reviews},
        updated_reviews,
        graph_exec_id,
        is_copilot,
    )

    # CoPilot sessions resume when the LLM retries run_block, not from here.
    if not is_copilot and updated_reviews:
        await _resume_if_nothing_pending(
            user_id,
            graph_exec_id,
            updated_reviews,
            organization_id=organization_id,
            team_id=team_id,
        )

    return ReviewOutcome(
        graph_exec_id=graph_exec_id,
        approved_count=sum(
            1 for r in updated_reviews.values() if r.status == ReviewStatus.APPROVED
        ),
        rejected_count=sum(
            1 for r in updated_reviews.values() if r.status == ReviewStatus.REJECTED
        ),
        auto_approval_failed_count=failed,
    )


def _one_execution(reviews, graph_exec_id: Optional[str]) -> str:
    """The single run every decision in the request belongs to."""
    graph_exec_ids = {review.graph_exec_id for review in reviews}
    if graph_exec_id is not None:
        outside = graph_exec_ids - {graph_exec_id}
        if outside:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review(s) not found for run {graph_exec_id}",
            )
        return graph_exec_id

    if len(graph_exec_ids) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="All reviews in a single request must belong to the same execution.",
        )
    return next(iter(graph_exec_ids))


async def _assert_awaiting_review(user_id: str, graph_exec_id: str) -> None:
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
            detail=(
                "Cannot process reviews while execution status is "
                f"{graph_exec_meta.status}"
            ),
        )


async def _record_auto_approvals(
    user_id: str,
    requested: dict[str, bool],
    updated_reviews: dict[str, Any],
    graph_exec_id: str,
    is_copilot: bool,
) -> int:
    """Save one auto-approval per node, returning how many could not be saved."""
    node_exec_ids = [
        node_exec_id
        for node_exec_id, review in updated_reviews.items()
        if review.status == ReviewStatus.APPROVED and requested.get(node_exec_id, False)
    ]
    node_id_map = await _resolve_node_ids(node_exec_ids, graph_exec_id, is_copilot)

    # Deduplicate by node_id — concurrent reviews of one node would otherwise race.
    by_node: dict[str, Any] = {}
    for node_exec_id in node_exec_ids:
        node_id = node_id_map.get(node_exec_id)
        if node_id and node_id not in by_node:
            by_node[node_id] = updated_reviews[node_exec_id]

    results = await asyncio.gather(
        *[
            _create_auto_approval(user_id, node_id, review)
            for node_id, review in by_node.items()
        ],
        return_exceptions=True,
    )
    return sum(1 for saved in results if isinstance(saved, BaseException) or not saved)


async def _create_auto_approval(user_id: str, node_id: str, review) -> bool:
    try:
        await create_auto_approval_record(
            user_id=user_id,
            graph_exec_id=review.graph_exec_id,
            graph_id=review.graph_id,
            graph_version=review.graph_version,
            node_id=node_id,
            payload=review.payload,
        )
        return True
    except Exception as e:
        logger.error(
            f"Failed to create auto-approval record for node {node_id}", exc_info=e
        )
        return False


async def _resolve_node_ids(
    node_exec_ids: list[str], graph_exec_id: str, is_copilot: bool
) -> dict[str, str]:
    """Resolve node_exec_id -> node_id for auto-approval records.

    CoPilot synthetic IDs encode node_id in the format "{node_id}:{random}".
    """
    if not node_exec_ids:
        return {}
    if is_copilot:
        return {neid: parse_node_id_from_exec_id(neid) for neid in node_exec_ids}

    node_execs = await get_node_executions(
        graph_exec_id=graph_exec_id, include_exec_data=False
    )
    by_exec_id = {ne.node_exec_id: ne.node_id for ne in node_execs}
    for missing in [neid for neid in node_exec_ids if neid not in by_exec_id]:
        logger.error(f"Failed to resolve node_id for {missing}: execution not found.")
    return {neid: by_exec_id[neid] for neid in node_exec_ids if neid in by_exec_id}


async def _resume_if_nothing_pending(
    user_id: str,
    graph_exec_id: str,
    updated_reviews: dict[str, Any],
    *,
    organization_id: Optional[str],
    team_id: Optional[str],
) -> None:
    if await has_pending_reviews_for_graph_exec(graph_exec_id):
        return

    first_review = next(iter(updated_reviews.values()))
    try:
        user = await get_user_by_id(user_id)
        settings = await get_graph_settings(
            user_id=user_id, graph_id=first_review.graph_id
        )
        workspace = await get_or_create_workspace(user_id)
        await add_graph_execution(
            graph_id=first_review.graph_id,
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            execution_context=ExecutionContext(
                human_in_the_loop_safe_mode=settings.human_in_the_loop_safe_mode,
                sensitive_action_safe_mode=settings.sensitive_action_safe_mode,
                user_timezone=(
                    user.timezone if user.timezone != USER_TIMEZONE_NOT_SET else "UTC"
                ),
                workspace_id=workspace.id,
            ),
            organization_id=organization_id,
            team_id=team_id,
        )
        logger.info(f"Resumed execution {graph_exec_id}")
    except Exception as e:
        logger.error(f"Failed to resume execution {graph_exec_id}: {e}")
