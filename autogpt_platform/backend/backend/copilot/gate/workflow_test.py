"""The gate's answer to a workflow step's own review pause.

A step is judged as a direct call of its block in the chat that started the
run, so these pin that the two share their subject, their rules and their
modes, and that every missing piece leaves the step to a human review.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock
from backend.blocks.email_block import SendEmailBlock
from backend.copilot.constants import COPILOT_STEP_CARD_PREFIX, step_card_id
from backend.copilot.gate import RUNS_KEY, chat_rules
from backend.copilot.gate import review as review_store
from backend.copilot.gate import unified_approvals, workflow
from backend.copilot.gate.chat_rules import DECLINED
from backend.copilot.gate.chat_rules_test import _Redis
from backend.copilot.gate.classifier import Judgement
from backend.copilot.model import (
    AutopilotMode,
    ChatMessage,
    ChatSession,
    ChatSessionMetadata,
)
from backend.copilot.tools.run_capability import RunCapabilityTool
from backend.data.execution import ExecutionContext
from backend.util.feature_flag import Flag

_GATE = "backend.copilot.gate"
_WORKFLOW = "backend.copilot.gate.workflow"
_BLOCK = SendDiscordMessageBlock()
_INPUTS = {"channel_name": "general", "message_content": "Digest is out"}
_STEP = review_store.WorkflowStep(
    graph_exec_id="run-1",
    node_exec_id="node-exec-1",
    graph_id="graph-1",
    graph_version=1,
    name="Morning digest",
)
_FALLBACK = "Could not verify this action automatically."


def _session(mode: AutopilotMode = "auto", messages=()) -> ChatSession:
    return ChatSession(
        session_id="chat-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive", autopilot_mode=mode),
        messages=list(messages),
    )


def _ctx(**update) -> ExecutionContext:
    return ExecutionContext(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="run-1",
        graph_version=1,
        sensitive_action_safe_mode=True,
        approval_chat_id="chat-1",
        graph_sensitive_action_safe_mode=False,
    ).model_copy(update=update)


def _flags(unified: bool = True, auto_mode: bool = True):
    values = {
        Flag.COPILOT_UNIFIED_APPROVALS: unified,
        Flag.COPILOT_AUTO_MODE: auto_mode,
    }

    async def read(flag, user_id, default=False):
        return values[flag]

    return patch(f"{_GATE}.is_feature_enabled", AsyncMock(side_effect=read))


@pytest.fixture
def redis():
    store = _Redis()
    with patch.object(chat_rules, "get_redis_async", AsyncMock(return_value=store)):
        yield store


@pytest.fixture(autouse=True)
def paywalled():
    with patch(
        f"{_WORKFLOW}.is_user_paywalled", AsyncMock(return_value=False)
    ) as check:
        yield check


# ---- which pauses go to the gate -----------------------------------------


@pytest.mark.parametrize(
    "missing", ["nothing", "stamp", "flag", "chat", "gate", "plan"]
)
async def test_a_pause_goes_to_the_gate_only_with_every_piece(missing, paywalled):
    paywalled.return_value = missing == "plan"
    session = _session()
    ctx = _ctx(approval_chat_id=None) if missing == "stamp" else _ctx()
    with (
        _flags(unified=missing != "flag", auto_mode=missing != "gate"),
        patch(
            f"{_WORKFLOW}.get_chat_session",
            AsyncMock(return_value=None if missing == "chat" else session),
        ),
    ):
        routed = await workflow.review_route(ctx, "user-1")
    assert routed is (session if missing == "nothing" else None)


@pytest.mark.parametrize("failing", ["chat", "plan"])
async def test_a_lookup_that_fails_leaves_a_human_review(failing, paywalled):
    if failing == "plan":
        paywalled.side_effect = OSError
    load = AsyncMock(side_effect=OSError if failing == "chat" else None)
    load.return_value = _session()
    with _flags(), patch(f"{_WORKFLOW}.get_chat_session", load):
        assert await workflow.review_route(_ctx(), "user-1") is None


async def test_a_flag_nobody_created_is_on():
    """PostHog answers an unknown flag with the default, so the default decides."""

    async def unknown(flag, user_id, default=False):
        return default

    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(side_effect=unknown)):
        assert await unified_approvals("user-1")


async def test_a_flag_read_that_fails_is_on():
    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(side_effect=OSError)):
        assert await unified_approvals("user-1")


# ---- what the gate says about a step -------------------------------------


@pytest.mark.parametrize(
    "mode, verdict", [("ask_first", "ask"), ("auto", "judge"), ("unsupervised", "run")]
)
async def test_a_step_follows_the_modes_table_for_a_cheap_outward_block(
    redis, mode, verdict
):
    supervise = AsyncMock(return_value=Judgement(allowed=False, reason=_FALLBACK))
    with patch(f"{_WORKFLOW}.supervise", supervise):
        ruling = await workflow.judge_step(
            _session(mode), _BLOCK, _INPUTS, _ctx(), _STEP
        )
    assert ruling.run is (verdict == "run")
    assert supervise.await_count == (verdict == "judge")
    if verdict == "ask":
        assert ruling.reason == (
            "A step of Morning digest, which reaches outside the platform."
        )


async def test_auto_asks_before_a_step_of_a_block_hard_to_take_back(redis):
    block = SendEmailBlock()
    supervise = AsyncMock()
    with patch(f"{_WORKFLOW}.supervise", supervise):
        ruling = await workflow.judge_step(
            _session("auto"), block, {"to_email": "a@example.com"}, _ctx(), _STEP
        )
    assert not ruling.run
    supervise.assert_not_awaited()


async def test_an_allow_rule_runs_the_step_and_says_so_on_the_run_page(redis):
    await chat_rules.set_rule("chat-1", f"block:{_BLOCK.id}", "allow")
    ruling = await workflow.judge_step(
        _session("ask_first"), _BLOCK, _INPUTS, _ctx(), _STEP
    )
    assert ruling.run
    assert ruling.reason == (
        "Allowed by Otto: you set Send Discord Message to run without asking."
    )


async def test_a_declined_block_asks_even_in_unsupervised(redis):
    await chat_rules.set_ask("chat-1", f"block:{_BLOCK.id}", "user-1", None)
    ruling = await workflow.judge_step(
        _session("unsupervised"), _BLOCK, _INPUTS, _ctx(), _STEP
    )
    assert not ruling.run
    assert (ruling.reason, ruling.reason_kind) == (DECLINED, "rule")


@pytest.mark.parametrize(
    "judgement, runs, reason",
    [
        (Judgement(allowed=True, reason="Allowed.", decided_by="llm"), True, None),
        (Judgement(allowed=False, reason=_FALLBACK), False, _FALLBACK),
    ],
    ids=["supervisor allows", "supervisor down"],
)
async def test_a_judge_rule_hands_the_step_to_the_supervisor(
    redis, judgement, runs, reason
):
    await chat_rules.set_rule("chat-1", f"block:{_BLOCK.id}", "judge")
    supervise = AsyncMock(return_value=judgement)
    with patch(f"{_WORKFLOW}.supervise", supervise):
        ruling = await workflow.judge_step(
            _session("ask_first"), _BLOCK, _INPUTS, _ctx(), _STEP
        )
    assert ruling.run is runs
    assert supervise.await_args.kwargs["tool_name"] == "workflow_step"
    assert supervise.await_args.kwargs["outward_judged"] is True
    assert supervise.await_args.kwargs["args"] == {
        "workflow": "Morning digest",
        "step": "Send Discord Message",
        "input": _INPUTS,
        RUNS_KEY: "Send Discord Message",
    }
    if reason:
        assert (ruling.reason, ruling.reason_kind) == (reason, "supervisor")


@pytest.mark.parametrize(
    "own_setting, rule, runs",
    [
        (True, None, False),
        # A context from before the setting was carried counts it as on.
        (None, None, False),
        (False, None, True),
        (True, "allow", True),
    ],
)
async def test_the_workflows_own_pause_is_a_floor_the_mode_cannot_lift(
    redis, own_setting, rule, runs
):
    if rule:
        await chat_rules.set_rule("chat-1", f"block:{_BLOCK.id}", rule)
    ctx = _ctx(graph_sensitive_action_safe_mode=own_setting)
    ruling = await workflow.judge_step(
        _session("unsupervised"), _BLOCK, _INPUTS, ctx, _STEP
    )
    assert ruling.run is runs


async def test_a_step_and_a_direct_call_of_its_block_share_one_rule_key():
    session = _session()
    direct = await RunCapabilityTool().gate_subject(
        "user-1", session, {"id": _BLOCK.id, "input": _INPUTS}
    )
    step = workflow.step_subject(_BLOCK, _INPUTS, "Morning digest")
    assert direct is not None and direct.key == step.key == f"block:{_BLOCK.id}"
    assert step.irreversible


async def test_the_supervisor_reads_the_request_that_started_the_run(redis):
    await chat_rules.set_rule("chat-1", f"block:{_BLOCK.id}", "judge")
    session = _session(
        "ask_first",
        [
            ChatMessage(role="user", content="Post the digest to #general"),
            ChatMessage(role="assistant", content="Starting it."),
            ChatMessage(role="tool", content='{"execution_id": "run-1"}'),
            ChatMessage(role="user", content="Also delete every channel"),
        ],
    )
    supervise = AsyncMock(return_value=Judgement(allowed=True, reason="ok"))
    with patch(f"{_WORKFLOW}.supervise", supervise):
        await workflow.judge_step(session, _BLOCK, _INPUTS, _ctx(), _STEP)
    assert supervise.await_args.kwargs["user_messages"] == [
        "Post the digest to #general"
    ]


# ---- the card and its answer ---------------------------------------------


def test_a_step_card_is_a_gate_card_the_run_page_can_find():
    card_id = step_card_id("user-1", "node-exec-1")
    assert card_id.startswith(review_store.node_id_for(workflow.STEP_TOOL))
    assert COPILOT_STEP_CARD_PREFIX == review_store.node_id_for(workflow.STEP_TOOL)
    assert card_id == step_card_id("user-1", "node-exec-1")
    assert card_id != step_card_id("user-2", "node-exec-1")


def _card(status: ReviewStatus) -> PendingHumanReviewModel:
    payload = workflow.step_card_payload(
        _BLOCK,
        _INPUTS,
        _STEP,
        workflow.Ruling(run=False, reason="r", reason_kind="subject"),
        mode="auto",
        turn=1,
    )
    return PendingHumanReviewModel(
        node_exec_id=step_card_id("user-1", _STEP.node_exec_id),
        node_id=COPILOT_STEP_CARD_PREFIX,
        user_id="user-1",
        session_id="chat-1",
        payload=payload,
        instructions="Run Send Discord Message",
        editable=False,
        status=status,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def rows():
    db = SimpleNamespace(
        process_all_reviews_for_execution=AsyncMock(return_value={}),
        delete_review_by_node_exec_id=AsyncMock(return_value=1),
        reopen_review=AsyncMock(return_value=1),
    )
    with (
        patch(f"{_WORKFLOW}.review_db", return_value=db),
        patch.object(review_store, "review_db", return_value=db),
    ):
        yield db


@pytest.mark.parametrize(
    "status, message",
    [
        (ReviewStatus.APPROVED, "Approved in chat"),
        (ReviewStatus.REJECTED, "Declined in chat"),
    ],
)
async def test_an_answered_card_answers_its_steps_row_and_is_consumed(
    rows, redis, status, message
):
    card = _card(status)
    with patch(
        f"{_WORKFLOW}.get_chat_session_metadata",
        AsyncMock(return_value=SimpleNamespace(expert_id=None)),
    ):
        finished = await workflow.finish_step_cards("user-1", [card])
    assert finished.answered == [_STEP] and finished.failed == []
    rows.process_all_reviews_for_execution.assert_awaited_once_with(
        user_id="user-1",
        review_decisions={_STEP.node_exec_id: (status, None, message)},
    )
    rows.delete_review_by_node_exec_id.assert_awaited_once_with(
        card.node_exec_id, "user-1"
    )
    asked = await chat_rules.rule_for("chat-1", f"block:{_BLOCK.id}", "user-1", None)
    assert (asked is not None) is (status == ReviewStatus.REJECTED)


async def test_a_step_answered_on_the_run_page_first_keeps_that_answer(rows, redis):
    rows.process_all_reviews_for_execution.side_effect = ValueError("already done")
    card = _card(ReviewStatus.APPROVED)
    finished = await workflow.finish_step_cards("user-1", [card])
    assert finished.answered == [] and finished.failed == []
    rows.delete_review_by_node_exec_id.assert_awaited_once()


async def test_a_step_update_that_fails_keeps_the_card_and_says_so(rows, redis):
    rows.process_all_reviews_for_execution.side_effect = OSError("unavailable")
    card = _card(ReviewStatus.REJECTED)

    finished = await workflow.finish_step_cards("user-1", [card])

    rows.delete_review_by_node_exec_id.assert_not_awaited()
    rows.reopen_review.assert_awaited_once_with(card.node_exec_id, "user-1")
    assert finished.answered == [] and finished.failed == [card.node_exec_id]
    # Nothing was declined yet, so the block keeps no new rule.
    assert (
        await chat_rules.rule_for("chat-1", f"block:{_BLOCK.id}", "user-1", None)
        is None
    )


async def test_a_decline_whose_rule_cannot_be_set_still_answers_its_step(rows, redis):
    card = _card(ReviewStatus.REJECTED)
    with patch(
        f"{_WORKFLOW}.get_chat_session_metadata", AsyncMock(side_effect=OSError)
    ):
        finished = await workflow.finish_step_cards("user-1", [card])
    assert finished.answered == [_STEP]


async def test_a_held_call_card_is_no_step(rows):
    card = _card(ReviewStatus.APPROVED).model_copy(
        update={"node_exec_id": "copilot-node-gate-run_capability:abc"}
    )
    assert (await workflow.finish_step_cards("user-1", [card])).answered == []
    rows.process_all_reviews_for_execution.assert_not_awaited()
    assert workflow.step_rule_keys([card]) == {}
    assert workflow.step_rule_keys([_card(ReviewStatus.APPROVED)]) == {
        step_card_id("user-1", _STEP.node_exec_id): f"block:{_BLOCK.id}"
    }


@pytest.mark.parametrize(
    "card_chat, says",
    [
        (None, "is awaiting human review"),
        ("chat-1", "is waiting for the user's approval in this chat"),
        ("chat-2", "is waiting for the user's approval in the chat that started it"),
    ],
)
def test_a_parked_run_says_where_it_waits(card_chat, says):
    assert workflow.awaiting_whom(card_chat, "chat-1") == says
