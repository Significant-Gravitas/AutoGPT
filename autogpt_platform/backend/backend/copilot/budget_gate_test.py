from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.budget_gate import (
    AutoContinueGate,
    gate_auto_continue,
    record_refusal,
    refusal_for,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
)
from backend.copilot.turn_pause import TurnPause, continues_tainted, pause_of

_MOD = "backend.copilot.budget_gate"
WEEKLY_RESET = datetime(2026, 10, 19, tzinfo=UTC)


@pytest.fixture
def allowance():
    """The rate-limit reads; tests set what ``check_rate_limit`` does."""
    with (
        patch(
            f"{_MOD}.get_global_rate_limits",
            AsyncMock(return_value=(3_000_000, 6_000_000, SubscriptionTier.PRO)),
        ) as limits,
        patch(f"{_MOD}.check_rate_limit", AsyncMock()) as check,
    ):
        yield limits, check


def _usage_state(tier: SubscriptionTier):
    return patch(
        f"{_MOD}.get_ready_usage_state",
        AsyncMock(return_value=MagicMock(tier=tier)),
    )


def _session() -> ChatSession:
    session = ChatSession.new("u-1", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="summarise my inbox", sequence=0),
        ChatMessage(role="assistant", content="Done.", sequence=1),
    ]
    return session


@pytest.mark.asyncio
async def test_a_turn_with_allowance_left_runs(allowance):
    _, check = allowance
    assert await refusal_for("u-1") is None
    check.assert_awaited_once_with(
        user_id="u-1",
        daily_cost_limit=3_000_000,
        weekly_cost_limit=6_000_000,
        expected_tier=SubscriptionTier.PRO,
    )


@pytest.mark.asyncio
async def test_a_turn_without_a_user_is_not_checked(allowance):
    limits, _ = allowance
    assert await refusal_for(None) is None
    limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_used_up_window_refuses_the_turn_until_it_refills(allowance):
    _, check = allowance
    check.side_effect = RateLimitExceeded("weekly", WEEKLY_RESET)
    assert await refusal_for("u-1") == TurnPause(
        reason="weekly",
        resets_at=WEEKLY_RESET,
        checkpoint_saved=False,
        started=False,
    )


@pytest.mark.asyncio
async def test_a_spent_trial_refuses_the_turn_with_no_reset(allowance):
    # The trial's end rides on the exception, but it is not a refill.
    _, check = allowance
    check.side_effect = RateLimitExceeded("trial", WEEKLY_RESET)
    pause = await refusal_for("u-1")
    assert pause is not None
    assert (pause.reason, pause.resets_at, pause.started) == ("trial", None, False)


@pytest.mark.asyncio
async def test_a_window_that_allows_nothing_refuses_with_no_reset(allowance):
    # No plan, or a trial that is over: zero limits never refill on their own,
    # so the pause can't offer to pick the task up at the next reset.
    limits, check = allowance
    limits.return_value = (0, 0, SubscriptionTier.NO_TIER)
    check.side_effect = RateLimitExceeded("weekly", WEEKLY_RESET)
    with _usage_state(SubscriptionTier.NO_TIER):
        pause = await refusal_for("u-1")
    assert pause is not None
    assert (pause.reason, pause.resets_at, pause.started) == ("weekly", None, False)


@pytest.mark.asyncio
async def test_an_ended_trial_is_refused_without_reading_its_stale_tier(allowance):
    # get_global_rate_limits reports an ended trial as NO_TIER with no
    # allowance while its usage state still says TRIAL; check_rate_limit's
    # tier guard would call that unavailable and fail the turn with "try
    # again in a moment" instead of saying it can't run.
    limits, check = allowance
    limits.return_value = (0, 0, SubscriptionTier.NO_TIER)
    check.side_effect = RateLimitUnavailable("Subscription changed")
    with _usage_state(SubscriptionTier.TRIAL):
        pause = await refusal_for("u-1")
    # Named as the trial, which is what ended: Upgrade, not a reset.
    assert pause == TurnPause(
        reason="trial", resets_at=None, checkpoint_saved=False, started=False
    )
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_allowance_is_raised_for_the_caller(allowance):
    _, check = allowance
    check.side_effect = RateLimitUnavailable()
    with pytest.raises(RateLimitUnavailable):
        await refusal_for("u-1")


@pytest.mark.asyncio
async def test_a_refusal_is_saved_as_a_marker_row():
    session = _session()
    pause = TurnPause(reason="daily", started=False, checkpoint_saved=False)
    with patch(f"{_MOD}.upsert_chat_session", AsyncMock()) as save:
        await record_refusal(session, pause)
    save.assert_awaited_once_with(session)
    assert pause_of(session.messages[-1]) == pause


@pytest.mark.asyncio
@pytest.mark.parametrize("tainted", [True, False])
async def test_a_refusal_keeps_the_refused_turns_taint(tainted: bool):
    """For the resume at the budget reset, which picks this pause up."""
    session = _session()
    pause = TurnPause(reason="daily", started=False, checkpoint_saved=False)
    with patch(f"{_MOD}.upsert_chat_session", AsyncMock()):
        await record_refusal(session, pause, tainted=tainted)
    assert continues_tainted(session.messages) is tainted


@pytest.mark.asyncio
async def test_a_failed_save_does_not_fail_the_refusal():
    session = _session()
    pause = TurnPause(reason="daily", started=False, checkpoint_saved=False)
    with patch(f"{_MOD}.upsert_chat_session", AsyncMock(side_effect=OSError)):
        await record_refusal(session, pause)
    assert pause_of(session.messages[-1]) == pause


@pytest.mark.asyncio
async def test_a_chat_already_ending_in_a_marker_gets_no_second_one():
    session = _session()
    first = TurnPause(reason="daily", started=False, checkpoint_saved=False)
    with patch(f"{_MOD}.upsert_chat_session", AsyncMock()) as save:
        await record_refusal(session, first)
        await record_refusal(session, first.model_copy(update={"reason": "weekly"}))
    save.assert_awaited_once()
    assert pause_of(session.messages[-1]) == first


@pytest.fixture
def queued():
    with patch(f"{_MOD}.peek_pending_count", AsyncMock(return_value=2)) as peek:
        yield peek


@pytest.mark.asyncio
async def test_a_chatgpt_turn_continues_unchecked(allowance, queued):
    limits, _ = allowance
    gate = await gate_auto_continue(_session(), "u-1", platform_billed=False)
    assert gate == AutoContinueGate(proceed=True)
    queued.assert_not_awaited()
    limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_nothing_queued_needs_no_check(allowance, queued):
    limits, _ = allowance
    queued.return_value = 0
    gate = await gate_auto_continue(_session(), "u-1", platform_billed=True)
    assert gate == AutoContinueGate(proceed=True)
    limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_messages_continue_while_the_allowance_lasts(allowance, queued):
    gate = await gate_auto_continue(_session(), "u-1", platform_billed=True)
    assert gate == AutoContinueGate(proceed=True)


@pytest.mark.asyncio
async def test_queued_messages_wait_when_the_allowance_is_used_up(allowance, queued):
    _, check = allowance
    check.side_effect = RateLimitExceeded("weekly", WEEKLY_RESET)
    session = _session()
    with patch(f"{_MOD}.upsert_chat_session", AsyncMock()) as save:
        gate = await gate_auto_continue(session, "u-1", platform_billed=True)

    assert not gate.proceed
    assert gate.pause is not None and not gate.pause.started
    assert (gate.pause.reason, gate.pause.resets_at) == ("weekly", WEEKLY_RESET)
    # Recorded in the chat; the queue itself was only looked at, never drained.
    assert pause_of(session.messages[-1]) == gate.pause
    save.assert_awaited_once_with(session)
    queued.assert_awaited_once_with(session.session_id)


@pytest.mark.asyncio
async def test_queued_messages_wait_when_the_allowance_is_unreadable(allowance, queued):
    _, check = allowance
    check.side_effect = RateLimitUnavailable()
    session = _session()
    gate = await gate_auto_continue(session, "u-1", platform_billed=True)
    assert gate == AutoContinueGate(proceed=False)
    assert pause_of(session.messages[-1]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tainted", [True, False])
async def test_a_refused_continuation_keeps_the_turns_taint(
    allowance, queued, tainted: bool
):
    _, check = allowance
    check.side_effect = RateLimitExceeded("weekly", WEEKLY_RESET)
    session = _session()
    with patch(f"{_MOD}.upsert_chat_session", AsyncMock()):
        await gate_auto_continue(session, "u-1", platform_billed=True, tainted=tainted)
    assert continues_tainted(session.messages) is tainted
