from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.at_reset import (
    FireVerdict,
    NoResetAhead,
    at_reset_schedule_id,
    cancel_resume_at_reset_in_background,
    fire_verdict,
    next_reset_run_at,
    reset_jitter_seconds,
)
from backend.copilot.continuation import KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE
from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage, ChatSessionInfo, ChatSessionMetadata
from backend.copilot.pending_message_helpers import StreamRegistryUnavailable
from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
    UsageWindow,
)

_MOD = "backend.copilot.at_reset"
SESSION = "5b0c2c0e-9b1f-4c6e-8f43-0d7e2a1b3c4d"
DAILY_RESET = datetime(2026, 10, 15, tzinfo=UTC)
WEEKLY_RESET = datetime(2026, 10, 19, tzinfo=UTC)
SET_AT = datetime(2026, 10, 14, 15, 0, tzinfo=UTC)
JITTER = timedelta(seconds=reset_jitter_seconds(SESSION))


def _status(daily_used: int, weekly_used: int, *, daily_limit: int = 3_000_000):
    return CoPilotUsageStatus(
        daily=UsageWindow(used=daily_used, limit=daily_limit, resets_at=DAILY_RESET),
        weekly=UsageWindow(used=weekly_used, limit=6_000_000, resets_at=WEEKLY_RESET),
        tier=SubscriptionTier.PRO,
    )


def _row(role: str, content: str, written: datetime | None) -> ChatMessage:
    return ChatMessage(role=role, content=content, created_at=written)


def _chat(*rows: ChatMessage, provider: str = "platform") -> PaginatedMessages:
    session = ChatSessionInfo(
        session_id=SESSION,
        user_id="user-1",
        usage=[],
        started_at=SET_AT - timedelta(hours=2),
        updated_at=SET_AT,
        metadata=ChatSessionMetadata.model_validate({"llm_auth_provider": provider}),
    )
    return PaginatedMessages(
        messages=list(rows), has_more=False, oldest_sequence=0, session=session
    )


@pytest.fixture
def limits():
    """The budget reads; tests set the usage ``status`` and the gate."""
    with (
        patch(
            f"{_MOD}.get_global_rate_limits",
            AsyncMock(return_value=(3_000_000, 6_000_000, SubscriptionTier.PRO)),
        ) as global_limits,
        patch(f"{_MOD}.get_usage_status", AsyncMock()) as status,
        patch(f"{_MOD}.check_rate_limit", AsyncMock()) as gate,
    ):
        yield global_limits, status, gate


@pytest.fixture
def chat_reads():
    """Nothing running in the chat, and no user message beyond the window."""
    store = MagicMock(get_latest_user_message_in_session=AsyncMock(return_value=None))
    with (
        patch(f"{_MOD}.chat_db", return_value=store),
        patch(f"{_MOD}.is_turn_in_flight", AsyncMock(return_value=False)) as busy,
    ):
        yield store, busy


@pytest.fixture
def trial(limits):
    """A trial user on a legacy offer: daily and weekly caps under a lifetime
    allowance with most of it left. Tests spend it as they need."""
    global_limits, _, _ = limits
    global_limits.return_value = (3_000_000, 6_000_000, SubscriptionTier.TRIAL)
    state = MagicMock(active=True, cost_microdollars=4_000_000)
    state.offer.total_cost_limit = 10_000_000
    store = MagicMock(get_subscription_trial=AsyncMock(return_value=state))
    with patch(f"{_MOD}.credit_db", return_value=store):
        yield state


def test_one_id_per_kind_per_chat():
    assert at_reset_schedule_id("resume", SESSION) == f"at-reset-resume-{SESSION}"
    assert at_reset_schedule_id("send", SESSION) == f"at-reset-send-{SESSION}"


def test_jitter_is_stable_per_chat_and_inside_ten_minutes():
    offsets = {reset_jitter_seconds(f"session-{i}") for i in range(200)}
    assert all(0 <= offset < 600 for offset in offsets)
    # Spread out, not everyone on the same second.
    assert len(offsets) > 100
    assert reset_jitter_seconds(SESSION) == reset_jitter_seconds(SESSION)


@pytest.mark.asyncio
async def test_out_of_todays_budget_waits_for_tomorrow(limits):
    _, status, _ = limits
    status.return_value = _status(daily_used=3_000_000, weekly_used=1_000_000)
    run_at = await next_reset_run_at("user-1", SESSION, "send")
    assert run_at == DAILY_RESET + timedelta(seconds=120) + JITTER


@pytest.mark.asyncio
async def test_out_of_both_waits_for_the_one_that_resets_last(limits):
    _, status, _ = limits
    status.return_value = _status(daily_used=3_000_000, weekly_used=6_500_000)
    run_at = await next_reset_run_at("user-1", SESSION, "send")
    assert run_at == WEEKLY_RESET + timedelta(seconds=120) + JITTER
    after_reset = run_at - WEEKLY_RESET - timedelta(seconds=120)
    assert timedelta(0) <= after_reset < timedelta(seconds=600)


@pytest.mark.asyncio
async def test_unreadable_usage_is_unavailable_not_a_budget_back(limits):
    # Read as no usage, an outage would tell the client to send right away.
    _, status, _ = limits
    status.side_effect = RateLimitUnavailable("Redis down")
    with pytest.raises(RateLimitUnavailable):
        await next_reset_run_at("user-1", SESSION, "send")
    assert status.await_args.kwargs["fail_closed"] is True


@pytest.mark.asyncio
async def test_a_resume_fires_a_minute_after_a_send_on_the_same_chat(limits):
    _, status, _ = limits
    status.return_value = _status(daily_used=3_000_000, weekly_used=0)
    send = await next_reset_run_at("user-1", SESSION, "send")
    resume = await next_reset_run_at("user-1", SESSION, "resume")
    assert resume - send == timedelta(seconds=60)


@pytest.mark.asyncio
async def test_nothing_exhausted_means_nothing_to_wait_for(limits):
    _, status, _ = limits
    status.return_value = _status(daily_used=2_999_999, weekly_used=0)
    with pytest.raises(NoResetAhead) as raised:
        await next_reset_run_at("user-1", SESSION, "resume")
    assert raised.value.reason == "not_out_of_budget"


@pytest.mark.asyncio
async def test_an_uncapped_window_is_never_exhausted(limits):
    _, status, _ = limits
    status.return_value = _status(daily_used=9_000_000, weekly_used=0, daily_limit=-1)
    with pytest.raises(NoResetAhead) as raised:
        await next_reset_run_at("user-1", SESSION, "send")
    assert raised.value.reason == "not_out_of_budget"


@pytest.mark.asyncio
async def test_a_spent_trial_allowance_never_resets(limits, trial):
    trial.cost_microdollars = 10_000_000
    with pytest.raises(NoResetAhead) as raised:
        await next_reset_run_at("user-1", SESSION, "resume")
    assert raised.value.reason == "no_reset"
    limits[1].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "daily_used, weekly_used, reset",
    [(3_000_000, 2_000_000, DAILY_RESET), (1_000_000, 6_000_000, WEEKLY_RESET)],
    ids=["daily-window", "weekly-window"],
)
async def test_a_trials_own_window_resets_like_any_other(
    limits, trial, daily_used, weekly_used, reset
):
    # The lifetime allowance has room; only the offer's tighter window is out.
    limits[1].return_value = _status(daily_used=daily_used, weekly_used=weekly_used)
    run_at = await next_reset_run_at("user-1", SESSION, "send")
    assert run_at == reset + timedelta(seconds=120) + JITTER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowance",
    [(0, 0, SubscriptionTier.NO_TIER), (0, 6_000_000, SubscriptionTier.PRO)],
    ids=["lapsed-trial-or-plan", "a-window-that-allows-nothing"],
)
async def test_a_window_with_no_allowance_never_resets(limits, allowance):
    global_limits, status, _ = limits
    global_limits.return_value = allowance
    with pytest.raises(NoResetAhead) as raised:
        await next_reset_run_at("user-1", SESSION, "send")
    assert raised.value.reason == "no_reset"
    status.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        _row("user", "never mind, do this instead", SET_AT + timedelta(minutes=5)),
        _row("assistant", KEEP_GOING_NOTE, SET_AT + timedelta(minutes=5)),
        _row("assistant", RESUMED_AT_RESET_NOTE, SET_AT + timedelta(hours=9)),
    ],
)
async def test_a_resume_stands_down_when_the_chat_was_picked_up(
    limits, chat_reads, row
):
    verdict = await fire_verdict("resume", _chat(row), SET_AT)
    assert verdict.action == "drop"
    limits[2].assert_not_awaited()


@pytest.mark.asyncio
async def test_a_newer_user_message_further_back_still_counts(limits, chat_reads):
    store, _ = chat_reads
    store.get_latest_user_message_in_session.return_value = _row(
        "user", "carry on", SET_AT + timedelta(minutes=1)
    )
    verdict = await fire_verdict("resume", _chat(), SET_AT)
    assert verdict.action == "drop"


@pytest.mark.asyncio
async def test_older_rows_and_unstamped_rows_dont_supersede_a_resume(
    limits, chat_reads
):
    chat = _chat(
        _row("user", "build the report", SET_AT - timedelta(minutes=30)),
        _row("assistant", KEEP_GOING_NOTE, SET_AT - timedelta(minutes=20)),
        _row("assistant", "Here is where I got to.", SET_AT + timedelta(minutes=1)),
        _row("user", "no timestamp", None),
    )
    verdict = await fire_verdict("resume", chat, SET_AT)
    assert verdict == FireVerdict(action="fire")


@pytest.mark.asyncio
async def test_a_send_fires_whatever_the_user_said_since(limits, chat_reads):
    chat = _chat(_row("user", "later message", SET_AT + timedelta(minutes=5)))
    verdict = await fire_verdict("send", chat, SET_AT)
    assert verdict.action == "fire"


@pytest.mark.asyncio
async def test_a_busy_chat_tries_again(limits, chat_reads):
    _, busy = chat_reads
    busy.return_value = True
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "retry"
    busy.side_effect = StreamRegistryUnavailable()
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "retry"


@pytest.mark.asyncio
async def test_a_chat_off_the_platform_route_skips_the_budget(limits, chat_reads):
    verdict = await fire_verdict("resume", _chat(provider="codex"), SET_AT)
    assert verdict.action == "fire"
    limits[2].assert_not_awaited()


@pytest.mark.asyncio
async def test_a_resume_on_a_chat_moved_to_microsoft_365_is_dropped(limits, chat_reads):
    """It would hand Microsoft the resume note as the user's message."""
    chat = _chat(provider="microsoft_365_copilot")
    verdict = await fire_verdict("resume", chat, SET_AT)
    assert verdict.action == "drop"
    assert verdict.why == "the chat now runs on Microsoft 365 Copilot"


@pytest.mark.asyncio
async def test_a_send_on_a_microsoft_365_chat_still_goes(limits, chat_reads):
    chat = _chat(provider="microsoft_365_copilot")
    verdict = await fire_verdict("send", chat, SET_AT)
    assert verdict.action == "fire"


@pytest.mark.asyncio
async def test_still_out_of_budget_waits_for_the_next_reset(limits, chat_reads):
    _, status, gate = limits
    gate.side_effect = RateLimitExceeded("daily", DAILY_RESET)
    status.return_value = _status(daily_used=3_000_000, weekly_used=6_000_000)
    verdict = await fire_verdict("resume", _chat(), SET_AT)
    assert verdict.action == "wait"
    assert verdict.run_at == WEEKLY_RESET + timedelta(seconds=180) + JITTER


@pytest.mark.asyncio
async def test_an_unreadable_budget_tries_again(limits, chat_reads):
    _, _, gate = limits
    gate.side_effect = RateLimitUnavailable()
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "retry"


@pytest.mark.asyncio
async def test_an_account_with_no_allowance_drops_the_turn_when_it_fires(
    limits, chat_reads
):
    # An ended trial reads as NO_TIER with no allowance while its usage
    # state still says TRIAL: the tier guard would report that as
    # unavailable, and the turn would be retried for an hour.
    global_limits, _, gate = limits
    global_limits.return_value = (0, 0, SubscriptionTier.NO_TIER)
    gate.side_effect = RateLimitUnavailable("Subscription changed")
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "drop"
    gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_spent_trial_drops_the_turn_when_it_fires(limits, chat_reads, trial):
    limits[2].side_effect = RateLimitExceeded("trial", SET_AT)
    trial.cost_microdollars = 10_000_000
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "drop"


@pytest.mark.asyncio
async def test_a_trials_spent_day_waits_for_tomorrow(limits, chat_reads, trial):
    limits[2].side_effect = RateLimitExceeded("daily", DAILY_RESET)
    limits[1].return_value = _status(daily_used=3_000_000, weekly_used=2_000_000)
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "wait"
    assert verdict.run_at == DAILY_RESET + timedelta(seconds=120) + JITTER


@pytest.mark.asyncio
async def test_cancelling_a_resume_runs_in_the_background_and_spares_a_send():
    client = MagicMock(cancel_at_reset_turn=AsyncMock(return_value=True))
    with patch(f"{_MOD}.get_scheduler_client", return_value=client) as factory:
        await cancel_resume_at_reset_in_background(SESSION, "user-1")
    client.cancel_at_reset_turn.assert_awaited_once_with(
        user_id="user-1", session_id=SESSION, kind="resume"
    )
    # A cancel lost on a connection the scheduler had just closed is repeated.
    factory.assert_called_once_with(should_retry=True)


@pytest.mark.asyncio
async def test_a_failing_cancel_is_swallowed():
    client = MagicMock(cancel_at_reset_turn=AsyncMock(side_effect=RuntimeError("down")))
    with patch(f"{_MOD}.get_scheduler_client", return_value=client):
        task = cancel_resume_at_reset_in_background(SESSION, "user-1")
        await task
    assert task.exception() is None
