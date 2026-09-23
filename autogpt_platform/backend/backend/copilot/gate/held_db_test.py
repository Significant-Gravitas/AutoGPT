"""Held calls against Postgres and Redis: the queue, and what an answer runs.

The call under test is a stub under ``post_to_chat_platform``'s name, so the
gate treats it as the outward action it is (Auto asks) while the test sees
exactly what the approval ran it with.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.copilot.gate import chat_rules, check_action, held
from backend.copilot.gate import review as review_store
from backend.copilot.model import (
    AutopilotMode,
    ChatMessage,
    ChatSession,
    append_and_save_message,
    get_chat_session,
    update_session_autopilot_mode,
    upsert_chat_session,
)
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ResponseType, ToolResponseBase
from backend.data.db_accessors import review_db

_TOOL = "post_to_chat_platform"


class _Post(BaseTool):
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return _TOOL

    @property
    def description(self) -> str:
        return "posts"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        self.runs.append(kwargs)
        return ToolResponseBase(
            type=ResponseType.ERROR, message=f"posted {kwargs.get('text')}"
        )


@pytest.fixture
def gate_on():
    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def post_tool():
    tool = _Post()
    with patch("backend.copilot.tools.get_tool", return_value=tool):
        yield tool


async def _new_session(user_id: str, mode: AutopilotMode = "auto") -> ChatSession:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, mode)
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    return reloaded


async def _hold(session: ChatSession, user_id: str, text: str) -> str:
    decision = await check_action(
        _TOOL, {"text": text}, user_id, session, tool_call_id="call-1"
    )
    assert not decision.allowed and decision.review_id
    return decision.review_id


async def _answer(review_id: str, status: ReviewStatus, age=timedelta(0)) -> None:
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": review_id},
        data={"status": status, "reviewedAt": datetime.now(UTC) - age},
    )


async def _row(review_id: str, user_id: str):
    rows = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    return rows.get(review_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_two_held_calls_in_one_turn_both_wait(
    setup_test_user, test_user_id, gate_on
):
    session = await _new_session(test_user_id)

    first = await _hold(session, test_user_id, "one")
    second = await _hold(session, test_user_id, "two")

    waiting = await review_db().get_pending_reviews_for_execution(
        review_store.session_exec_id(session.session_id), test_user_id
    )
    assert [r.node_exec_id for r in waiting] == [first, second]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_retry_of_a_waiting_call_keeps_the_first_call(
    setup_test_user, test_user_id, gate_on
):
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "again")

    retry = await check_action(
        _TOOL, {"text": "again"}, test_user_id, session, tool_call_id="call-2"
    )
    await _answer(review_id, ReviewStatus.APPROVED)

    assert not retry.allowed and retry.review_id == review_id
    [call] = await held.answered(test_user_id, session.session_id)
    assert call.tool_call_id == "call-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failed_wake_is_not_repeated_for_the_same_cards(
    setup_test_user, test_user_id, gate_on
):
    """Even after an error reply, which lets an identical user row through."""
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "wake once")
    await _answer(review_id, ReviewStatus.APPROVED)
    dispatch = AsyncMock(side_effect=RuntimeError("queue down"))

    with patch("backend.copilot.executor.utils.dispatch_turn", dispatch):
        await held.wake(test_user_id, session.session_id)
        await append_and_save_message(
            session.session_id, ChatMessage(role="assistant", content="error")
        )
        await held.wake(test_user_id, session.session_id)

    reloaded = await get_chat_session(session.session_id, test_user_id)
    assert reloaded is not None
    wakes = [m for m in reloaded.messages if m.content == held.WAKE_MESSAGE]
    assert len(wakes) == 1
    assert dispatch.await_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approval_runs_the_call_with_its_stored_arguments(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Long enough that the card shows a clipped copy: the run gets the whole."""
    session = await _new_session(test_user_id)
    text = "hello team " + "x" * 5_000
    review_id = await _hold(session, test_user_id, text)
    await _answer(review_id, ReviewStatus.APPROVED)

    delivered = await held.resolve_answered(test_user_id, session)

    assert post_tool.runs == [{"text": text}]
    assert len(delivered) == 1
    assert "posted hello team" in delivered[0].content
    assert 'tool_call_id="call-1"' in delivered[0].content
    assert await _row(review_id, test_user_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_large_late_result_arrives_as_the_direct_result_would(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Longer than a typed follow-up may be: the engine caps it, nothing else."""
    session = await _new_session(test_user_id)
    text = "y" * 50_000
    with patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=False)
    ):
        direct = await post_tool.execute(test_user_id, session, "call-1", text=text)
    review_id = await _hold(session, test_user_id, text)
    await _answer(review_id, ReviewStatus.APPROVED)

    [delivered] = await held.resolve_answered(test_user_id, session)

    assert delivered.content == (
        f'<held_call_result tool="{_TOOL}" tool_call_id="call-1" '
        f'review_id="{review_id}">\n{direct.output}\n</held_call_result>'
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failure_after_the_claim_keeps_the_call_for_the_next_turn(
    setup_test_user, test_user_id, gate_on, post_tool
):
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "flaky")
    await _answer(review_id, ReviewStatus.APPROVED)

    with patch.object(held, "_outcome", AsyncMock(side_effect=RuntimeError("db"))):
        assert await held.resolve_answered(test_user_id, session) == []
    assert [
        c.review_id for c in await held.answered(test_user_id, session.session_id)
    ] == [review_id]

    [delivered] = await held.resolve_answered(test_user_id, session)
    assert "posted flaky" in delivered.content


@pytest.mark.parametrize(
    "status, expect",
    [(ReviewStatus.APPROVED, "posted keep"), (ReviewStatus.REJECTED, "declined")],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_failed_cap_still_delivers_the_real_outcome(
    setup_test_user, test_user_id, gate_on, post_tool, status, expect
):
    """A refusal stays a refusal, and a run is reported with its result."""
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "keep")
    await _answer(review_id, status)

    def broken_cap(_text: str) -> str:
        raise RuntimeError("cap failed")

    [delivered] = await held.resolve_answered(test_user_id, session, cap=broken_cap)

    assert expect in delivered.content
    assert "may have run" not in delivered.content
    assert post_tool.runs == (
        [{"text": "keep"}] if status == ReviewStatus.APPROVED else []
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failure_after_the_approval_was_spent_says_it_may_have_run(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Re-storing it would report "no longer open" for a call that reached the gate."""
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "spent")
    await _answer(review_id, ReviewStatus.APPROVED)

    async def spend_then_fail(user_id, *_args):
        await review_store.consume(review_id, user_id)
        raise RuntimeError("lost after the gate")

    with patch.object(held, "_outcome", spend_then_fail):
        [delivered] = await held.resolve_answered(test_user_id, session)

    assert "may have run" in delivered.content
    assert await held.answered(test_user_id, session.session_id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approval_run_with_the_flag_off_is_still_spent(
    setup_test_user, test_user_id, gate_on, post_tool
):
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "flag off")
    await _answer(review_id, ReviewStatus.APPROVED)

    with patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=False)
    ):
        await held.resolve_answered(test_user_id, session)

    assert post_tool.runs == [{"text": "flag off"}]
    assert await _row(review_id, test_user_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_row_approved_and_resolved_four_times_at_once_runs_once(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Four turns that all listed the answered card before any of them ran it."""
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "once")
    await _answer(review_id, ReviewStatus.APPROVED)
    listed = await held.answered(test_user_id, session.session_id)

    with patch.object(held, "answered", AsyncMock(return_value=listed)):
        results = await asyncio.gather(
            *(held.resolve_answered(test_user_id, session) for _ in range(4))
        )

    assert len(post_tool.runs) == 1
    assert sum(len(r) for r in results) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stale_approval_delivers_a_refusal_not_a_run(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Unsupervised by the time it resolves: a stale approval must still not run."""
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "late")
    await _answer(review_id, ReviewStatus.APPROVED, age=timedelta(hours=2))
    await update_session_autopilot_mode(
        session.session_id, test_user_id, "unsupervised"
    )
    session = await get_chat_session(session.session_id, test_user_id)
    assert session is not None

    delivered = await held.resolve_answered(test_user_id, session)

    assert post_tool.runs == []
    assert "expired" in delivered[0].content
    assert await _row(review_id, test_user_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_rejection_never_runs_and_the_tool_asks_from_then_on(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Even with the flag switched off since: the gate would then allow anything."""
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "no")
    await _answer(review_id, ReviewStatus.REJECTED)

    with patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=False)
    ):
        delivered = await held.resolve_answered(test_user_id, session)

    assert post_tool.runs == []
    assert "declined" in delivered[0].content
    assert await chat_rules.ask_reason(session.session_id, _TOOL) == chat_rules.DECLINED


@pytest.mark.asyncio(loop_scope="session")
async def test_a_held_call_returns_the_stub_and_no_partial_result(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """What a dependent call could read while its input is held: no result at all."""
    session = await _new_session(test_user_id)

    output = await post_tool.execute(test_user_id, session, "call-draft", text="draft")

    body = json.loads(str(output.output))
    assert body["type"] == "approval_required"
    assert body["review_id"]
    assert "posted" not in str(output.output)
    assert post_tool.runs == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_answer_on_an_idle_chat_starts_its_turn(
    setup_test_user, test_user_id, gate_on
):
    session = await _new_session(test_user_id)
    review_id = await _hold(session, test_user_id, "wake")
    dispatch = AsyncMock()
    with patch("backend.copilot.executor.utils.dispatch_turn", dispatch):
        await held.wake(test_user_id, session.session_id)
        dispatch.assert_not_awaited()

        await _answer(review_id, ReviewStatus.APPROVED)
        await held.wake(test_user_id, session.session_id)

    dispatch.assert_awaited_once()
    assert dispatch.await_args.kwargs["message"] == held.WAKE_MESSAGE
