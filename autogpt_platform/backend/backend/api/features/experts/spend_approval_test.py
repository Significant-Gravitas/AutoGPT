from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ReviewStatus

from backend.api.features.experts import spend_approval as sa
from backend.data.expert_spend import window_start
from backend.util.settings import Config

NEEDED = sa.SpendApprovalNeeded(
    expert_id="e-1", expert_name="Ada", spent=250, threshold=250, window="week"
)


def _arm(
    mocker,
    *,
    spent: int,
    threshold: int | None = 250,
    window: str = "week",
    flag: bool = True,
    approved: bool = False,
):
    mocker.patch.object(sa, "approval_threshold", return_value=threshold)
    mocker.patch.object(sa, "approval_window", return_value=window)
    mocker.patch.object(sa, "is_feature_enabled", AsyncMock(return_value=flag))
    get_spend = mocker.patch.object(sa, "get_spend", AsyncMock(return_value=spent))
    reviews = MagicMock(
        find_first=AsyncMock(return_value=MagicMock() if approved else None)
    )
    mocker.patch.object(
        prisma.models.PendingHumanReview, "prisma", return_value=reviews
    )
    expert = MagicMock()
    expert.name = "Ada"
    mocker.patch.object(
        prisma.models.Expert,
        "prisma",
        return_value=MagicMock(find_first=AsyncMock(return_value=expert)),
    )
    return get_spend, reviews


@pytest.mark.asyncio
@pytest.mark.parametrize("spent,expected", [(249, None), (250, NEEDED)])
async def test_check_fires_at_the_threshold_not_below(mocker, spent, expected) -> None:
    _arm(mocker, spent=spent)

    assert await sa.spend_approval_required("owner", "e-1") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold,flag", [(None, True), (250, False)])
async def test_disabled_check_reads_no_spend(mocker, threshold, flag) -> None:
    get_spend, _ = _arm(mocker, spent=1_000, threshold=threshold, flag=flag)

    assert await sa.spend_approval_required("owner", "e-1") is None
    get_spend.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("window", ["week", "day"])
async def test_an_approval_in_the_window_unlocks_it(mocker, window) -> None:
    get_spend, reviews = _arm(mocker, spent=999, window=window, approved=True)

    assert await sa.spend_approval_required("owner", "e-1") is None

    get_spend.assert_awaited_once_with("e-1", window)
    where = reviews.find_first.call_args.kwargs["where"]
    assert where["userId"] == "owner"
    assert where["nodeExecId"] == {"contains": "expert-spend:e-1:"}
    assert where["status"] == ReviewStatus.APPROVED
    assert where["reviewedAt"] == {"gte": window_start(window)}


def test_headline_names_the_expert_and_avoids_the_card_discriminator() -> None:
    assert NEEDED.headline == (
        "Ada has used 250 of 250 credits this week and needs your approval "
        "to keep spending"
    )
    # PendingReviewCard drops any instructions containing "Block".
    assert "Block" not in NEEDED.headline


def test_defaults_are_250_credits_per_iso_week() -> None:
    fields = Config.model_fields
    assert fields["expert_spend_approval_threshold_default"].default == 250
    assert fields["expert_spend_approval_window"].default == "week"


def test_spend_review_ids_are_recognised_on_both_rails() -> None:
    assert sa.is_spend_review(sa.spend_review_id("e-1", "exec-9"))
    assert sa.is_spend_review("copilot-node-expert-spend:e-1:abcd1234")
    assert not sa.is_spend_review("3f1a0c2e-node-exec")


def _park_mocks(mocker):
    create = mocker.patch.object(
        sa.human_review, "get_or_create_human_review", AsyncMock(return_value=None)
    )
    status = mocker.patch.object(
        sa, "update_graph_execution_stats", AsyncMock(return_value=MagicMock())
    )
    post = mocker.patch.object(
        sa.chat_db, "append_expert_run_message", AsyncMock(return_value="msg")
    )
    return create, status, post


@pytest.mark.asyncio
async def test_park_opens_a_review_on_the_run_and_sets_review_status(mocker) -> None:
    create, status, post = _park_mocks(mocker)

    await sa.park_execution_for_spend_approval(
        user_id="owner",
        graph_exec_id="exec-9",
        graph_id="graph",
        graph_version=3,
        needed=NEEDED,
        organization_id="org",
        team_id="team",
    )

    kwargs = create.await_args.kwargs
    assert kwargs["node_exec_id"] == "expert-spend:e-1:exec-9"
    assert kwargs["graph_exec_id"] == "exec-9"
    assert kwargs["message"] == NEEDED.headline
    assert kwargs["editable"] is False
    assert kwargs["input_data"]["spent_credits"] == 250
    assert kwargs["organization_id"] == "org"
    status.assert_awaited_once_with(
        graph_exec_id="exec-9", status=sa.ExecutionStatus.REVIEW
    )
    assert post.await_args.kwargs["expert_id"] == "e-1"


@pytest.mark.asyncio
async def test_a_park_that_cannot_hold_the_run_undoes_its_review_and_raises(
    mocker,
) -> None:
    """A waiting card over an execution that is not in REVIEW is worse than no
    gate: the resume path reads the durable status, so it would skip the check
    and requeue unapproved."""
    _, status, post = _park_mocks(mocker)
    status.return_value = None
    delete = mocker.patch.object(
        sa.human_review, "delete_review_by_node_exec_id", AsyncMock(return_value=1)
    )

    with pytest.raises(sa.SpendApprovalParkFailed):
        await sa.park_execution_for_spend_approval(
            user_id="owner",
            graph_exec_id="exec-9",
            graph_id="graph",
            graph_version=3,
            needed=NEEDED,
        )

    delete.assert_awaited_once_with("expert-spend:e-1:exec-9", "owner")
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_message_is_posted_once_per_window(mocker) -> None:
    _, _, post = _park_mocks(mocker)

    for exec_id in ("exec-1", "exec-2"):
        await sa.park_execution_for_spend_approval(
            user_id="owner",
            graph_exec_id=exec_id,
            graph_id="graph",
            graph_version=1,
            needed=NEEDED,
        )

    ids = {c.kwargs["message_id"] for c in post.await_args_list}
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_parked_decision_reads_the_run_review(mocker) -> None:
    review_id = sa.spend_review_id("e-1", "exec-9")
    lookup = mocker.patch.object(
        sa.human_review,
        "get_reviews_by_node_exec_ids",
        AsyncMock(return_value={review_id: MagicMock(status=ReviewStatus.REJECTED)}),
    )

    assert (
        await sa.parked_spend_decision("owner", "e-1", "exec-9")
        == ReviewStatus.REJECTED
    )
    lookup.assert_awaited_once_with([review_id], "owner")

    lookup.return_value = {}
    assert await sa.parked_spend_decision("owner", "e-1", "exec-9") is None


@pytest.mark.asyncio
async def test_chat_review_reuses_the_open_row_for_the_expert(mocker) -> None:
    existing = MagicMock(
        node_id="copilot-node-expert-spend:e-1",
        node_exec_id="copilot-node-expert-spend:e-1:abcd1234",
    )
    mocker.patch.object(
        sa.human_review,
        "get_pending_reviews_for_execution",
        AsyncMock(return_value=[existing]),
    )
    create = mocker.patch.object(
        sa.human_review, "get_or_create_human_review", AsyncMock()
    )

    review_id = await sa.open_chat_spend_review(
        user_id="owner", session_id="s-1", needed=NEEDED, block_name="Send Email"
    )

    assert review_id == existing.node_exec_id
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_review_opens_on_the_session_rails(mocker) -> None:
    mocker.patch.object(
        sa.human_review, "get_pending_reviews_for_execution", AsyncMock(return_value=[])
    )
    create = mocker.patch.object(
        sa.human_review, "get_or_create_human_review", AsyncMock()
    )

    review_id = await sa.open_chat_spend_review(
        user_id="owner", session_id="s-1", needed=NEEDED, block_name="Send Email"
    )

    kwargs = create.await_args.kwargs
    assert review_id == kwargs["node_exec_id"]
    assert review_id.startswith("copilot-node-expert-spend:e-1:")
    assert kwargs["graph_exec_id"] == "copilot-session-s-1"
    assert kwargs["editable"] is False
    assert kwargs["input_data"]["block"] == "Send Email"
