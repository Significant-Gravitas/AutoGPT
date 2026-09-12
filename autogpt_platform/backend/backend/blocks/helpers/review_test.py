"""Unit tests for the HITL review helper chain."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.helpers.review import HITLReviewHelper
from backend.data.human_review import ReviewResult, ReviewStatus


@pytest.mark.asyncio
async def test_handle_review_decision_threads_org_to_review_row():
    """The execution's org/team must reach the PendingHumanReview row —
    the review helper chain previously dropped them between the block's
    ExecutionContext and ``get_or_create_human_review``."""
    approved = ReviewResult(
        data={"x": 1},
        status=ReviewStatus.APPROVED,
        message="ok",
        processed=True,
        node_exec_id="ne-1",
    )
    get_or_create = AsyncMock(return_value=approved)

    with (
        patch.object(HITLReviewHelper, "check_approval", AsyncMock(return_value=None)),
        patch.object(HITLReviewHelper, "get_or_create_human_review", get_or_create),
    ):
        decision = await HITLReviewHelper.handle_review_decision(
            input_data={"x": 1},
            user_id="u1",
            node_id="n-1",
            node_exec_id="ne-1",
            graph_exec_id="ge-1",
            graph_id="g-1",
            graph_version=1,
            block_name="TestBlock",
            organization_id="org-exec",
            team_id="team-exec",
        )

    assert decision is not None
    assert decision.should_proceed is True
    kwargs = get_or_create.await_args.kwargs
    assert kwargs["organization_id"] == "org-exec"
    assert kwargs["team_id"] == "team-exec"
