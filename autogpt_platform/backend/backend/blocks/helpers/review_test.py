"""Unit tests for the HITL review helper chain."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock
from backend.blocks.helpers.review import HITLReviewHelper
from backend.copilot.gate import Ruling
from backend.data.execution import ExecutionContext, ExecutionStatus
from backend.data.human_review import ReviewResult, ReviewStatus
from backend.util.exceptions import BlockExecutionError


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


# ---- the chat's approval gate answers an irreversible step's pause --------

_WORKFLOW = "backend.copilot.gate.workflow"
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


def _pause(**helpers):
    """The block's own pause, with the review helper's DB calls stubbed."""
    stubs = {
        "check_approval": AsyncMock(return_value=None),
        "get_or_create_human_review": AsyncMock(return_value=None),
        "record_gate_approval": AsyncMock(return_value=_approved()),
        "update_node_execution_status": AsyncMock(),
        "update_review_processed_status": AsyncMock(),
        **helpers,
    }
    patches = [patch.object(HITLReviewHelper, k, v) for k, v in stubs.items()]
    return stubs, patches


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
