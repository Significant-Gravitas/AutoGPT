from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.at_reset import (
    FireVerdict,
    NoResetAhead,
    at_reset_schedule_id,
    cancel_resume_at_reset,
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
async def test_a_trial_never_resets(limits):
    global_limits, status, _ = limits
    global_limits.return_value = (5_000_000, 5_000_000, SubscriptionTier.TRIAL)
    with pytest.raises(NoResetAhead) as raised:
        await next_reset_run_at("user-1", SESSION, "resume")
    assert raised.value.reason == "no_reset"
    status.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_window_with_no_allowance_never_resets(limits):
    # A lapsed plan: the windows collapse to zero, and stay there.
    _, status, _ = limits
    status.return_value = _status(daily_used=0, weekly_used=0, daily_limit=0)
    with pytest.raises(NoResetAhead) as raised:
        await next_reset_run_at("user-1", SESSION, "send")
    assert raised.value.reason == "no_reset"


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
async def test_out_of_an_allowance_that_never_refills_drops_it(limits, chat_reads):
    global_limits, _, gate = limits
    gate.side_effect = RateLimitExceeded("trial", SET_AT)
    # Read at the gate as a paid tier, then as a trial once over it.
    global_limits.side_effect = [
        (3_000_000, 6_000_000, SubscriptionTier.PRO),
        (5_000_000, 5_000_000, SubscriptionTier.TRIAL),
    ]
    verdict = await fire_verdict("send", _chat(), SET_AT)
    assert verdict.action == "drop"


@pytest.mark.asyncio
async def test_cancelling_a_resume_is_best_effort():
    client = MagicMock(cancel_at_reset_turn=AsyncMock(return_value=True))
    with patch(f"{_MOD}.get_scheduler_client", return_value=client):
        await cancel_resume_at_reset(SESSION, "user-1")
    client.cancel_at_reset_turn.assert_awaited_once_with(
        user_id="user-1", session_id=SESSION, kind="resume"
    )
    client.cancel_at_reset_turn.side_effect = RuntimeError("scheduler down")
    with patch(f"{_MOD}.get_scheduler_client", return_value=client):
        await cancel_resume_at_reset(SESSION, "user-1")
