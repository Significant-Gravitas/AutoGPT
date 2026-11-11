"""Which clock an account runs on, and the Stripe re-read when our copy of a
billing period has ended."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.copilot import usage_clock
from backend.copilot.usage_clock import (
    NO_SUBSCRIPTION_RETRY_SECONDS,
    RECENT_STATE_MAX_USERS,
    REFRESH_RETRY_SECONDS,
    clock_from_state,
    forget_recent_states,
    needs_refresh,
    recent_state,
    remember_state,
    resolve_clock,
)
from backend.data.pro_activation import UsageActivationState
from backend.util.posthog_events import PostHogEvent

DAY = 86_400
CREATED = 1_750_000_000
START = 1_793_824_256  # Wed 4 Nov 2026 20:30:56 UTC
END = START + 30 * DAY
NOW = START + 10 * DAY


def _state(tier: str = "PRO", **fields) -> UsageActivationState:
    values = {
        "user_id": "u1",
        "tier": SubscriptionTier(tier),
        "ready": True,
        "created_at": CREATED,
        "has_stripe_customer": True,
        "period_start": START,
        "period_end": END,
        "billing_anchor": START,
    }
    return UsageActivationState(**{**values, **fields})


@pytest.fixture(autouse=True)
def _no_recent_states():
    forget_recent_states()
    yield
    forget_recent_states()


@pytest.fixture
def redis():
    client = MagicMock()
    client.set = AsyncMock(return_value=True)
    client.expire = AsyncMock()
    with patch.object(usage_clock, "get_redis_async", AsyncMock(return_value=client)):
        yield client


@pytest.fixture
def stripe_read():
    read = AsyncMock(return_value="saved")
    database = MagicMock(refresh_billing_clock=read)
    with patch.object(usage_clock, "credit_db", return_value=database):
        yield read


@pytest.fixture
def events():
    with patch.object(usage_clock.posthog_client, "capture") as capture:
        yield capture


@pytest.mark.parametrize("tier", ["BASIC", "PRO", "MAX", "BUSINESS"])
def test_a_stripe_billed_plan_runs_on_its_stripe_period(tier: str):
    clock = clock_from_state(_state(tier), NOW)

    assert clock is not None and clock.source == "stripe"
    # The second quarter of the billing month: 12 Nov 08:30:56.
    assert clock.windows.week_start == START + 7 * DAY + 12 * 3600


@pytest.mark.parametrize(
    "state",
    [
        _state("ENTERPRISE"),  # admin-managed, whatever Stripe row it once had
        _state("PRO", has_stripe_customer=False),  # granted by an admin
        _state("PRO", period_start=None, period_end=None),  # never copied
        _state("NO_TIER"),  # payment off: free use on the base limits
    ],
)
def test_everything_else_runs_monthly_from_the_day_the_account_was_created(
    state: UsageActivationState,
):
    clock = clock_from_state(state, NOW)

    assert clock is not None and clock.source == "account"
    assert (clock.windows.week_start - CREATED) % 3600 == 0  # same minute and second


def test_a_trial_runs_on_its_own_dates_in_seven_day_windows():
    trial = _state(
        "TRIAL",
        trial_started_at=NOW - 2 * DAY,
        trial_ends_at=NOW + 5 * DAY,
        period_start=None,
        period_end=None,
    )

    clock = clock_from_state(trial, NOW)

    assert clock is not None and clock.source == "trial"
    assert clock.windows.week_start == NOW - 2 * DAY
    assert clock.windows.week_end == NOW + 5 * DAY


def test_a_period_that_has_ended_is_held():
    clock = clock_from_state(_state(), END + 30)

    assert clock is not None and clock.source == "held"
    assert clock.windows.held


def test_a_state_from_the_previous_release_has_no_clock():
    old = UsageActivationState(user_id="u1", tier=SubscriptionTier.PRO, ready=True)

    assert clock_from_state(old, NOW) is None


@pytest.mark.parametrize(
    "state,now,expected",
    [
        (_state(), NOW, False),  # inside its period
        (_state(), END, True),  # the period has just ended
        (_state(period_start=None, period_end=None), NOW, True),  # never copied
        (_state("ENTERPRISE"), END + 5, False),
        (_state("TRIAL"), END + 5, False),
        (_state("NO_TIER"), END + 5, False),
        (_state(has_stripe_customer=False), END + 5, False),
    ],
)
def test_only_a_stripe_billed_plan_without_a_current_period_re_reads_stripe(
    state: UsageActivationState, now: int, expected: bool
):
    assert needs_refresh(state, now) is expected


@pytest.mark.asyncio
async def test_a_current_period_asks_stripe_nothing(redis, stripe_read):
    state, clock = await resolve_clock(_state(), NOW)

    assert clock is not None and clock.source == "stripe"
    redis.set.assert_not_awaited()
    stripe_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_ended_period_is_re_read_and_the_new_one_used(redis, stripe_read):
    renewed = _state(period_start=END, period_end=END + 31 * DAY)
    with patch.object(
        usage_clock, "get_ready_usage_state", AsyncMock(return_value=renewed)
    ):
        state, clock = await resolve_clock(_state(), END + 5)

    stripe_read.assert_awaited_once_with("u1")
    assert state == renewed
    assert clock is not None and clock.source == "stripe"
    assert clock.windows.week_start == END  # the new month starts straight away


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unchanged", "no_subscription"])
async def test_without_a_newer_period_the_last_window_is_held(
    redis, stripe_read, events, outcome: str
):
    stripe_read.return_value = outcome

    _, clock = await resolve_clock(_state(), END + 5)

    assert clock is not None and clock.source == "held"
    assert clock.windows.week_start == START + 22 * DAY + 12 * 3600  # window 4
    assert events.call_args.args[1] == PostHogEvent.USAGE_CLOCK_HELD
    assert events.call_args.args[2]["seconds_past_period_end"] == 5


@pytest.mark.asyncio
async def test_a_stripe_outage_holds_the_last_window_and_does_not_raise(
    redis, stripe_read, events
):
    stripe_read.side_effect = RuntimeError("stripe down")

    _, clock = await resolve_clock(_state(), END + 5)

    assert clock is not None and clock.source == "held"
    # The next read may come in a minute: the gate keeps its short life.
    redis.expire.assert_not_awaited()


@pytest.mark.asyncio
async def test_stripe_is_read_at_most_once_a_minute_per_user(redis, stripe_read):
    redis.set.return_value = None  # another check holds the gate

    _, clock = await resolve_clock(_state(), END + 5)

    assert clock is not None and clock.source == "held"
    stripe_read.assert_not_awaited()
    gate = redis.set.await_args
    assert gate.args[0] == "usage:clock_refresh:u1"
    assert gate.kwargs == {"nx": True, "ex": REFRESH_RETRY_SECONDS}


@pytest.mark.asyncio
async def test_no_subscription_is_remembered_for_six_hours(redis, stripe_read, events):
    """A paid tier on an account whose Stripe customer has no plan (a grant
    over an old customer, a cancel whose webhook was missed) is read once, not
    every minute, and runs on the account's clock meanwhile."""
    stripe_read.return_value = "no_subscription"
    unpaid = _state(period_start=None, period_end=None)

    _, clock = await resolve_clock(unpaid, NOW)

    assert clock is not None and clock.source == "account"
    redis.expire.assert_awaited_once_with(
        "usage:clock_refresh:u1", NO_SUBSCRIPTION_RETRY_SECONDS
    )
    assert events.call_args.args[1] == PostHogEvent.USAGE_CLOCK_MISSING
    assert events.call_count == 1


@pytest.mark.asyncio
async def test_with_redis_down_stripe_is_not_read_and_nothing_raises(stripe_read):
    with patch.object(
        usage_clock, "get_redis_async", AsyncMock(side_effect=ConnectionError())
    ):
        _, clock = await resolve_clock(_state(), END + 5)

    assert clock is not None and clock.source == "held"
    stripe_read.assert_not_awaited()


def test_a_recent_state_serves_recording_for_a_minute(monkeypatch):
    clock = MagicMock(return_value=1000.0)
    monkeypatch.setattr(usage_clock, "monotonic", clock)
    remember_state(_state())

    clock.return_value = 1059.0
    assert recent_state("u1", NOW) == _state()
    clock.return_value = 1061.0
    assert recent_state("u1", NOW) is None


def test_a_recent_state_is_never_used_past_its_billing_period():
    remember_state(_state())

    assert recent_state("u1", END - 1) is not None
    assert recent_state("u1", END) is None


def test_the_recent_states_are_bounded():
    for index in range(RECENT_STATE_MAX_USERS + 5):
        remember_state(_state(user_id=f"u{index}"))

    assert len(usage_clock._recent_states) == RECENT_STATE_MAX_USERS
    assert recent_state("u0", NOW) is None  # the oldest went first
    assert recent_state(f"u{RECENT_STATE_MAX_USERS + 4}", NOW) is not None
