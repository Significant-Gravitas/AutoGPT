"""Unit tests for the HITL review helper chain."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.basic import StoreValueBlock
from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock
from backend.blocks.google.docs import GoogleDocsShareBlock
from backend.blocks.helpers.review import SAFE_MODE_OFF_MESSAGE, HITLReviewHelper
from backend.blocks.medium import PublishToMediumBlock
from backend.copilot.gate import Ruling
from backend.data.execution import ExecutionContext, ExecutionStatus
from backend.data.human_review import ReviewResult, ReviewStatus
from backend.util.exceptions import (
    BlockExecutionError,
    BlockInputError,
    BlockUnknownError,
)
from backend.util.test import execute_block_test


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
        _stored_reviews({}),
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


# ---- the chat's approval gate answers an irreversible step's pause --------

_WORKFLOW = "backend.copilot.gate.workflow"
_REVIEW = "backend.blocks.helpers.review"
_STEP = SimpleNamespace(
    graph_exec_id="ge-1",
    node_exec_id="ne-1",
    graph_id="g-1",
    graph_version=1,
    name="Morning digest",
)
_INPUT = {"channel_name": "general", "message_content": "hi"}


def _approved(message: str = "ok") -> ReviewResult:
    return ReviewResult(
        data=_INPUT,
        status=ReviewStatus.APPROVED,
        message=message,
        processed=True,
        node_exec_id="ne-1",
    )


def _stored_reviews(
    rows: dict[str, SimpleNamespace], read_error: Exception | None = None
):
    """The reviews already stored, by step, as the Prisma-or-RPC read returns
    them, or ``read_error`` from that read."""
    reads = SimpleNamespace(
        get_reviews_by_node_exec_ids=AsyncMock(
            return_value=rows, side_effect=read_error
        )
    )
    return patch(f"{_REVIEW}.review_db", return_value=reads)


def _pause(
    rows: dict[str, SimpleNamespace] | None = None,
    read_error: Exception | None = None,
    **helpers,
):
    """The block's own pause, with the review helper's DB calls stubbed and
    ``rows`` the reviews already stored."""
    stubs = {
        "check_approval": AsyncMock(return_value=None),
        "get_or_create_human_review": AsyncMock(return_value=None),
        "record_gate_approval": AsyncMock(return_value=_approved()),
        "update_node_execution_status": AsyncMock(),
        "update_review_processed_status": AsyncMock(),
        **helpers,
    }
    patches = [patch.object(HITLReviewHelper, k, v) for k, v in stubs.items()]
    return stubs, [*patches, _stored_reviews(rows or {}, read_error)]


async def _run(
    stubs,
    patches,
    ruling=None,
    routed=True,
    card_opens=True,
    judge_error=None,
    card_error=None,
):
    ctx = ExecutionContext(
        user_id="u1",
        graph_id="g-1",
        graph_exec_id="ge-1",
        graph_version=1,
        sensitive_action_safe_mode=True,
        approval_chat_id="chat-1",
    )
    judge = AsyncMock(
        return_value=ruling or Ruling(run=False, reason="asks"),
        side_effect=judge_error,
    )
    open_card = AsyncMock(return_value=card_opens, side_effect=card_error)
    chat = SimpleNamespace(session_id="chat-1") if routed else None
    with (
        patch(f"{_WORKFLOW}.review_route", AsyncMock(return_value=chat)),
        patch(f"{_WORKFLOW}.workflow_step", AsyncMock(return_value=_STEP)),
        patch(f"{_WORKFLOW}.judge_step", judge),
        patch(f"{_WORKFLOW}.open_step_card", open_card),
    ):
        for p in patches:
            p.start()
        try:
            result = await SendDiscordMessageBlock().is_block_exec_need_review(
                _INPUT,
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
                execution_context=ctx,
            )
        finally:
            for p in patches:
                p.stop()
    return result, judge, open_card


async def test_a_step_the_gate_does_not_answer_is_a_human_review():
    stubs, patches = _pause()
    (paused, _), judge, _ = await _run(stubs, patches, routed=False)
    assert paused
    judge.assert_not_awaited()
    assert "alert" not in stubs["get_or_create_human_review"].await_args.kwargs


async def test_a_gate_yes_records_the_approval_and_the_step_runs():
    stubs, patches = _pause()
    ruling = Ruling(run=True, reason="Allowed by Otto: x.", by="mode")
    (paused, data), _, open_card = await _run(stubs, patches, ruling=ruling)
    assert not paused and data == _INPUT
    assert stubs["record_gate_approval"].await_args.kwargs["message"] == ruling.reason
    stubs["get_or_create_human_review"].assert_not_awaited()
    open_card.assert_not_awaited()


async def test_a_gate_yes_cannot_overturn_a_decline_already_on_the_row():
    declined = _approved().model_copy(update={"status": ReviewStatus.REJECTED})
    stubs, patches = _pause(record_gate_approval=AsyncMock(return_value=declined))
    with pytest.raises(BlockExecutionError):
        await _run(stubs, patches, ruling=Ruling(run=True, by="mode"))


async def test_a_gate_ask_parks_the_step_silently_behind_its_card():
    stubs, patches = _pause()
    (paused, _), _, open_card = await _run(stubs, patches)
    assert paused
    open_card.assert_awaited_once()
    assert stubs["get_or_create_human_review"].await_count == 1
    assert stubs["get_or_create_human_review"].await_args.kwargs["alert"] is False
    status = stubs["update_node_execution_status"].await_args.kwargs["status"]
    assert status == ExecutionStatus.REVIEW


@pytest.mark.parametrize("card_error", [None, OSError])
async def test_a_card_that_cannot_open_leaves_an_alerted_human_review(card_error):
    stubs, patches = _pause()
    (paused, _), _, _ = await _run(
        stubs, patches, card_opens=False, card_error=card_error
    )
    assert paused
    calls = stubs["get_or_create_human_review"].await_args_list
    assert [c.kwargs.get("alert", True) for c in calls] == [False, True]


async def test_a_gate_that_fails_leaves_a_human_review():
    stubs, patches = _pause()
    (paused, _), _, open_card = await _run(stubs, patches, judge_error=OSError)
    assert paused
    open_card.assert_not_awaited()
    assert "alert" not in stubs["get_or_create_human_review"].await_args.kwargs


async def test_a_human_in_the_loop_pause_never_reaches_the_gate():
    """Only an irreversible block's own pause passes the block; a question
    the workflow's author put to a person stays theirs."""
    route = AsyncMock()
    _, patches = _pause()
    with patch(f"{_WORKFLOW}.review_route", route):
        for p in patches:
            p.start()
        try:
            await HITLReviewHelper.handle_review_decision(
                input_data=_INPUT,
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
            )
        finally:
            for p in patches:
                p.stop()
    route.assert_not_awaited()


async def test_a_sub_graphs_step_names_its_own_run_not_its_parents():
    """A sub-graph's context is copied from its parent run, so the step's own
    ids come from the pause, never from the context."""
    stubs, patches = _pause()
    ctx = ExecutionContext(
        user_id="u1",
        graph_id="parent-graph",
        graph_exec_id="parent-run",
        graph_version=7,
        sensitive_action_safe_mode=True,
        approval_chat_id="chat-1",
    )
    judge = AsyncMock(return_value=Ruling(run=True, reason="ok", by="mode"))
    graphs = SimpleNamespace(
        get_graph_metadata=AsyncMock(return_value=SimpleNamespace(name="Child"))
    )
    with (
        patch(
            f"{_WORKFLOW}.review_route",
            AsyncMock(return_value=SimpleNamespace(session_id="chat-1")),
        ),
        patch(f"{_WORKFLOW}.judge_step", judge),
        patch(f"{_WORKFLOW}.graph_db", return_value=graphs),
    ):
        for p in patches:
            p.start()
        try:
            await SendDiscordMessageBlock().is_block_exec_need_review(
                _INPUT,
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
                execution_context=ctx,
            )
        finally:
            for p in patches:
                p.stop()
    step = judge.await_args.args[4]
    assert (step.graph_exec_id, step.graph_id, step.graph_version) == ("ge-1", "g-1", 1)
    graphs.get_graph_metadata.assert_awaited_once_with("g-1", 1)
    recorded = stubs["record_gate_approval"].await_args.kwargs
    assert (recorded["graph_exec_id"], recorded["graph_id"]) == ("ge-1", "g-1")


# ---- the order of a step's decision ---------------------------------------

_EDITED = {"channel_name": "general", "message_content": "edited"}
_APPROVE_FUTURE = ReviewResult(
    data=_INPUT,
    status=ReviewStatus.APPROVED,
    message="Auto-approved (user approved all future actions for this node)",
    processed=True,
    node_exec_id="auto_approve_ge-1_n-1",
)


@pytest.mark.parametrize("ask", [True, False], ids=["safe_mode_on", "safe_mode_off"])
@pytest.mark.parametrize(
    "stored, processed, outcome",
    [
        (None, False, "runs"),
        (ReviewStatus.WAITING, False, "waits"),
        (ReviewStatus.APPROVED, False, "runs_approved"),
        (ReviewStatus.APPROVED, True, "runs_approved"),
        (ReviewStatus.REJECTED, False, "stops"),
        (ReviewStatus.REJECTED, True, "stops"),
    ],
    ids=[
        "no_review",
        "waiting",
        "approved",
        "approved_processed",
        "rejected",
        "rejected_processed",
    ],
)
async def test_a_steps_own_review_decides_it_whatever_the_safe_mode(
    stored: ReviewStatus | None, processed: bool, outcome: str, ask: bool
):
    """The step's own review comes first, whatever the safe mode: a rejection
    stops it, an approval runs the reviewer's data, and a review still waiting
    holds it for the answer. An "approve future" for the node never outranks
    it; with no review, the step runs as it is."""
    rows = (
        {}
        if stored is None
        else {
            "ne-1": SimpleNamespace(
                payload=_EDITED,
                status=stored,
                review_message="",
                processed=processed,
            )
        }
    )
    stubs, patches = _pause(
        rows, check_approval=AsyncMock(return_value=_APPROVE_FUTURE)
    )
    for p in patches:
        p.start()
    try:
        decision = await HITLReviewHelper.handle_review_decision(
            input_data=_INPUT,
            user_id="u1",
            node_id="n-1",
            node_exec_id="ne-1",
            graph_exec_id="ge-1",
            graph_id="g-1",
            graph_version=1,
            ask=ask,
        )
    finally:
        for p in patches:
            p.stop()

    if outcome == "waits":
        assert decision is None
        stubs["update_node_execution_status"].assert_awaited_once_with(
            exec_id="ne-1", status=ExecutionStatus.REVIEW
        )
        stubs["check_approval"].assert_not_awaited()
    else:
        assert decision is not None
        assert decision.should_proceed is (outcome != "stops")
        if outcome != "stops":
            assert decision.review_result.data == (
                _EDITED if outcome == "runs_approved" else _INPUT
            )
        if stored is None:
            assert decision.message == (
                _APPROVE_FUTURE.message if ask else SAFE_MODE_OFF_MESSAGE
            )
    assert stubs["check_approval"].await_count == int(ask and stored is None)
    stubs["get_or_create_human_review"].assert_not_awaited()
    marked = stubs["update_review_processed_status"]
    if outcome in ("runs_approved", "stops") and not processed:
        marked.assert_awaited_once_with(node_exec_id="ne-1", processed=True)
    else:
        marked.assert_not_awaited()


@pytest.mark.parametrize("ask", [True, False], ids=["safe_mode_on", "safe_mode_off"])
async def test_an_answer_that_cannot_be_read_fails_the_step(ask: bool):
    """A failed read is never taken for no answer, which would run a step its
    reviewer may have rejected."""
    stubs, patches = _pause(read_error=ConnectionError("down"))
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError, match="review saved for step ne-1"):
            await HITLReviewHelper.handle_review_decision(
                input_data=_INPUT,
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
                ask=ask,
            )
    finally:
        for p in patches:
            p.stop()
    stubs["check_approval"].assert_not_awaited()


def test_an_approval_never_replaces_the_runs_credentials():
    """The reviewer's edits apply over this run's input, which keeps the
    credentials this run resolved; a stored credential field never wins."""
    own = {
        **_INPUT,
        "credentials": {
            "id": "this-runs-login",
            "provider": "discord",
            "type": "api_key",
            "title": "Bot",
        },
    }
    reviewed = {
        **_INPUT,
        "message_content": "edited",
        "credentials": {
            "id": "another-login",
            "provider": "discord",
            "type": "api_key",
            "title": "Other",
        },
    }

    approved = SendDiscordMessageBlock()._approved_input(own, reviewed)

    assert approved == {**own, "message_content": "edited"}


def _login(login_id: str, provider: str = "discord") -> dict:
    return {"id": login_id, "provider": provider, "type": "api_key", "title": "x"}


_POST = {"title": "Launch", "content": "Body", "publish_status": "bogus"}


async def test_a_step_nobody_reviewed_runs_on_its_own_input_untouched():
    """With the safe mode off and no review, the gate hands the run's own input
    back as it is, never coerced again: a value coercion can't convert stays
    for the block's own input check, not an error from the gate."""
    own = {**_POST, "credentials": _login("this-runs-login", "medium")}
    _, patches = _pause()
    for p in patches:
        p.start()
    try:
        paused, data = await PublishToMediumBlock().is_block_exec_need_review(
            own,
            user_id="u1",
            node_id="n-1",
            node_exec_id="ne-1",
            graph_exec_id="ge-1",
            graph_id="g-1",
            graph_version=1,
            execution_context=ExecutionContext(
                user_id="u1", graph_exec_id="ge-1", sensitive_action_safe_mode=False
            ),
        )
    finally:
        for p in patches:
            p.stop()

    assert not paused
    assert data == own


def test_an_approved_value_that_cannot_be_converted_is_an_input_error():
    own = {**_POST, "publish_status": "draft"}
    own["credentials"] = _login("this-runs-login", "medium")

    with pytest.raises(BlockInputError, match="publish_status"):
        PublishToMediumBlock()._approved_input(own, dict(_POST))


async def test_a_step_whose_saved_review_cannot_be_read_does_not_run():
    """Run as the executor runs it, the step fails with an error that says
    its saved review could not be read, and its own run never starts."""
    block = SendDiscordMessageBlock()
    run = AsyncMock()
    setattr(block, "run", run)
    with _stored_reviews({}, ConnectionError("down")):
        with pytest.raises(BlockUnknownError, match="review saved for step ne-1"):
            async for _ in block.execute(
                dict(_INPUT),
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
                execution_context=ExecutionContext(
                    user_id="u1", graph_exec_id="ge-1", sensitive_action_safe_mode=False
                ),
            ):
                pass
    run.assert_not_called()


def test_a_field_the_reviewer_removed_stays_removed():
    """Clearing a field on the review card drops it from the payload: the
    approved step runs without it, not with the run's original value."""
    own = {**_INPUT, "server_name": "Ops", "credentials": _login("this-runs-login")}

    approved = SendDiscordMessageBlock()._approved_input(own, dict(_INPUT))

    assert approved == {**_INPUT, "credentials": own["credentials"]}


def test_a_reviewer_never_sets_a_secret_or_a_login():
    """A secret and a login are the run's own: an edited one, or one the
    reviewer added, never reaches the approved step."""
    content = {"title": "Launch", "content": "Body", "publish_status": "draft"}
    own = {**content, "author_id": "this-runs-author"}
    own["credentials"] = _login("this-runs-login", "medium")
    reviewed = {**content, "title": "Edited", "author_id": "another-author"}
    reviewed["credentials"] = _login("another-login", "medium")
    no_secret = {k: v for k, v in own.items() if k != "author_id"}

    edited = PublishToMediumBlock()._approved_input(own, reviewed)
    added = PublishToMediumBlock()._approved_input(no_secret, reviewed)

    assert edited == {**own, "title": "Edited"}
    assert added == {**no_secret, "title": "Edited"}


@pytest.mark.parametrize("reviewer_key", ["_credentials_id", "credentials_id"])
@pytest.mark.parametrize("own_key", ["_credentials_id", "credentials_id", None])
def test_a_reviewer_picks_another_file_but_never_another_account(
    reviewer_key: str, own_key: str | None
):
    """The reviewer may pick another document; the account behind the picker
    is the run's own, under either spelling the picker model accepts. One the
    reviewer sends is dropped, and with no account of the run's, none."""
    own = {"id": "doc-1", "name": "Plan"}
    if own_key:
        own[own_key] = "this-runs-google"
    reviewed = {"id": "doc-2", "name": "Other", reviewer_key: "another-google"}

    approved = GoogleDocsShareBlock()._approved_input(
        {"document": own}, {"document": reviewed}
    )

    account = {own_key: "this-runs-google"} if own_key else {}
    assert approved == {"document": {"id": "doc-2", "name": "Other", **account}}


async def test_a_step_without_its_own_picker_account_never_runs_on_the_reviewers():
    """The run's picker carries no account and the approved payload names one:
    the step fails for missing credentials instead of running on that one."""
    block = GoogleDocsShareBlock()
    run = AsyncMock()
    setattr(block, "run", run)
    own = {"document": {"id": "doc-1", "name": "Plan"}, "email": "a@example.com"}
    reviewed = {
        "document": {"id": "doc-1", "name": "Plan", "credentials_id": "another"},
        "email": "a@example.com",
    }
    approved = SimpleNamespace(
        payload=reviewed,
        status=ReviewStatus.APPROVED,
        review_message="",
        processed=True,
    )
    _, patches = _pause({"ne-1": approved})
    for p in patches:
        p.start()
    try:
        with pytest.raises(BlockExecutionError, match="Missing credentials"):
            async for _ in block.execute(
                own,
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
                execution_context=ExecutionContext(
                    user_id="u1", graph_exec_id="ge-1", sensitive_action_safe_mode=True
                ),
            ):
                pass
    finally:
        for p in patches:
            p.stop()
    run.assert_not_called()


async def test_a_block_test_in_progress_does_not_hide_another_steps_rejection():
    """``execute_block_test`` skips the review gate only on the block it
    tests. While one is held mid-run, another step in the same process, with
    the safe mode off, still reads its stored rejection and does not run."""
    entered, release = asyncio.Event(), asyncio.Event()
    tested = StoreValueBlock()
    own_run = tested.run

    async def held_run(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=30)
        async for output in own_run(*args, **kwargs):
            yield output

    setattr(tested, "run", held_run)
    block_test = asyncio.create_task(execute_block_test(tested))
    rejected = SimpleNamespace(
        payload=_INPUT,
        status=ReviewStatus.REJECTED,
        review_message="Not this one",
        processed=False,
    )
    _, patches = _pause({"ne-1": rejected})
    for p in patches:
        p.start()
    try:
        await asyncio.wait_for(entered.wait(), timeout=30)
        with pytest.raises(BlockExecutionError, match="Not this one"):
            await SendDiscordMessageBlock().is_block_exec_need_review(
                _INPUT,
                user_id="u1",
                node_id="n-1",
                node_exec_id="ne-1",
                graph_exec_id="ge-1",
                graph_id="g-1",
                graph_version=1,
                execution_context=ExecutionContext(
                    user_id="u1", graph_exec_id="ge-1", sensitive_action_safe_mode=False
                ),
            )
    finally:
        for p in patches:
            p.stop()
        release.set()
        await asyncio.wait_for(block_test, timeout=30)
    assert "is_block_exec_need_review" not in vars(tested), "the stub outlived its test"
