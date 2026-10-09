"""A free trial has one lifetime budget, independent of calendar resets."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot import rate_limit
from backend.data import subscription_trial_config
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer

_USER = "free-trial-user"
_SUNDAY = datetime(2026, 10, 4, 23, 59, tzinfo=UTC)


@pytest.fixture
def clock(mocker: MockerFixture) -> MagicMock:
    clock = MagicMock(wraps=datetime)
    clock.now.return_value = _SUNDAY
    mocker.patch.object(rate_limit, "datetime", clock)
    mocker.patch.object(subscription_trial_config, "datetime", clock)
    return clock


@pytest.fixture
def trial(clock: MagicMock, mocker: MockerFixture) -> TrialState:
    trial = TrialState(
        id="trial-1",
        user_id=_USER,
        customer_id="cus_trial",
        offer=AcceptedTrialOffer(
            version="trial-budget-v2",
            new_users_from=_SUNDAY - timedelta(days=1),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=100_000_000,
            weekly_cost_limit=20_000_000,
            total_cost_limit=20_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=5000,
            currency="usd",
        ),
        checkout_session_id="cs_trial",
        subscription_id="sub_trial",
        checkout_attempt=1,
        success_url="https://example.com/success",
        cancel_url="https://example.com/cancel",
        checkout_metadata={},
        status="trialing",
        card_verified_at=_SUNDAY,
        started_at=_SUNDAY,
        ends_at=_SUNDAY + timedelta(days=7),
        consumed_at=_SUNDAY,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )
    store = MagicMock(get_subscription_trial=AsyncMock(return_value=trial))
    mocker.patch.object(rate_limit, "credit_db", return_value=store)
    for name in ("_fetch_user_tier", "get_user_tier"):
        mocker.patch.object(
            rate_limit, name, AsyncMock(return_value=rate_limit.SubscriptionTier.TRIAL)
        )
    return trial


@pytest.fixture
def usage(mocker: MockerFixture) -> dict[str, int]:
    counters: dict[str, int] = {}
    redis = AsyncMock()
    redis.get.side_effect = lambda key: str(counters.get(key, 0))
    mocker.patch.object(rate_limit, "get_redis_async", AsyncMock(return_value=redis))
    mocker.patch.object(
        rate_limit, "_fetch_cost_limits_flag", AsyncMock(return_value=None)
    )
    return counters


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", [0, 5_000_000, 19_999_999])
async def test_entire_trial_budget_can_be_spent_in_one_day(
    trial: TrialState, usage: dict[str, int], spent: int
):
    trial.cost_microdollars = spent
    usage[rate_limit._daily_key(_USER)] = spent
    usage[rate_limit._weekly_key(_USER)] = spent
    daily, weekly, tier = await rate_limit.get_global_rate_limits(
        _USER, config_daily=5_000_000, config_weekly=25_000_000
    )

    assert (daily, weekly, tier) == (
        100_000_000,
        20_000_000,
        rate_limit.SubscriptionTier.TRIAL,
    )
    await rate_limit.check_rate_limit(_USER, daily, weekly)
    remaining = await rate_limit.get_remaining_usd_budget(_USER, daily, weekly)

    assert remaining == pytest.approx((20_000_000 - spent) / 1_000_000)


@pytest.mark.asyncio
async def test_weekly_reset_does_not_replenish_trial_budget(
    trial: TrialState, usage: dict[str, int], clock: MagicMock
):
    trial.cost_microdollars = 15_000_000
    old_week_key = rate_limit._weekly_key(_USER)
    usage[rate_limit._daily_key(_USER)] = 15_000_000
    usage[old_week_key] = 15_000_000
    daily, weekly, _ = await rate_limit.get_global_rate_limits(_USER, 1, 1)
    before = await rate_limit.get_remaining_usd_budget(_USER, daily, weekly)

    clock.now.return_value = _SUNDAY + timedelta(minutes=2)
    assert rate_limit._weekly_key(_USER) != old_week_key
    assert usage.get(rate_limit._weekly_key(_USER), 0) == 0
    await rate_limit.check_rate_limit(_USER, daily, weekly)
    after = await rate_limit.get_remaining_usd_budget(_USER, daily, weekly)

    assert before == after == 5.0
    trial.cost_microdollars += 5_000_000
    usage[rate_limit._daily_key(_USER)] = 5_000_000
    usage[rate_limit._weekly_key(_USER)] = 5_000_000
    with pytest.raises(rate_limit.RateLimitExceeded) as exc:
        await rate_limit.check_rate_limit(_USER, daily, weekly)
    assert exc.value.window == "trial"
    assert await rate_limit.get_remaining_usd_budget(_USER, daily, weekly) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", [20_000_000, 25_000_000])
@pytest.mark.parametrize("windows_reset", [False, True])
async def test_exhausted_trial_stays_blocked_before_and_after_window_resets(
    trial: TrialState, usage: dict[str, int], spent: int, windows_reset: bool
):
    trial.cost_microdollars = spent
    if not windows_reset:
        usage[rate_limit._daily_key(_USER)] = spent
        usage[rate_limit._weekly_key(_USER)] = spent
    daily, weekly, _ = await rate_limit.get_global_rate_limits(_USER, 1, 1)

    with pytest.raises(rate_limit.RateLimitExceeded) as exc:
        await rate_limit.check_rate_limit(_USER, daily, weekly)

    assert exc.value.window == "trial"
    assert exc.value.resets_at == trial.ends_at
    assert await rate_limit.get_remaining_usd_budget(_USER, daily, weekly) == 0.0
    assert trial.cost_microdollars == spent
    assert trial.started_at == _SUNDAY
    assert trial.ends_at == _SUNDAY + timedelta(days=7)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier,expected_daily,expected_weekly",
    [
        (rate_limit.SubscriptionTier.BASIC, 4_000_000, 20_000_000),
        (rate_limit.SubscriptionTier.PRO, 5_000_000, 25_000_000),
    ],
)
async def test_paid_tiers_keep_their_daily_and_weekly_limits(
    clock: MagicMock,
    usage: dict[str, int],
    mocker: MockerFixture,
    tier: rate_limit.SubscriptionTier,
    expected_daily: int,
    expected_weekly: int,
):
    for name in ("_fetch_user_tier", "get_user_tier"):
        mocker.patch.object(rate_limit, name, AsyncMock(return_value=tier))
    mocker.patch.object(
        rate_limit, "_fetch_tier_multipliers_flag", AsyncMock(return_value=None)
    )
    store = mocker.patch.object(rate_limit, "credit_db")
    daily, weekly, resolved_tier = await rate_limit.get_global_rate_limits(
        _USER, 4_000_000, 20_000_000
    )
    assert (daily, weekly, resolved_tier) == (expected_daily, expected_weekly, tier)
    usage[rate_limit._daily_key(_USER)] = daily

    with pytest.raises(rate_limit.RateLimitExceeded) as exc:
        await rate_limit.check_rate_limit(_USER, daily, weekly)

    assert exc.value.window == "daily"
    usage[rate_limit._daily_key(_USER)] = 0
    usage[rate_limit._weekly_key(_USER)] = weekly
    with pytest.raises(rate_limit.RateLimitExceeded) as exc:
        await rate_limit.check_rate_limit(_USER, daily, weekly)
    assert exc.value.window == "weekly"
    store.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "daily_used,weekly_used", [(5_000_000, 25_000_000), (100_000_000, 100_000_000)]
)
async def test_spending_before_trial_cannot_block_remaining_trial_allowance(
    trial: TrialState, usage: dict[str, int], daily_used: int, weekly_used: int
):
    trial.cost_microdollars = 5_000_000
    usage[rate_limit._daily_key(_USER)] = daily_used
    usage[rate_limit._weekly_key(_USER)] = weekly_used
    before = dict(usage)
    daily, weekly, _ = await rate_limit.get_global_rate_limits(_USER, 1, 1)

    await rate_limit.check_rate_limit(_USER, daily, weekly)

    assert await rate_limit.get_remaining_usd_budget(_USER, daily, weekly) == 15.0
    assert usage == before


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", [5_000_000, 19_999_999, 20_000_000, 25_000_000])
async def test_public_trial_exhaustion_matches_lifetime_budget_across_weekly_reset(
    trial: TrialState, usage: dict[str, int], clock: MagicMock, spent: int
):
    trial.cost_microdollars = spent
    usage[rate_limit._daily_key(_USER)] = 100_000_000
    usage[rate_limit._weekly_key(_USER)] = 100_000_000
    before = dict(usage)
    daily, weekly, tier = await rate_limit.get_global_rate_limits(_USER, 1, 1)

    for now in (_SUNDAY, _SUNDAY + timedelta(minutes=2)):
        clock.now.return_value = now
        status = await rate_limit.get_usage_status(_USER, daily, weekly, tier=tier)
        public = rate_limit.CoPilotUsagePublic.from_status(status)

        assert status.weekly.used == spent
        assert status.weekly.limit == 20_000_000
        assert status.weekly.resets_at == trial.ends_at
        assert public.daily is not None and public.daily.percent_used < 100
        assert public.weekly is not None
        assert (public.weekly.percent_used >= 100) == (spent >= 20_000_000)
    assert usage == before


@pytest.mark.asyncio
async def test_legacy_trial_with_tighter_windows_retains_periodic_limits(
    trial: TrialState, usage: dict[str, int]
):
    trial.offer = trial.offer.model_copy(
        update={"daily_cost_limit": 2_000_000, "weekly_cost_limit": 10_000_000}
    )
    trial.cost_microdollars = 8_000_000
    usage[rate_limit._daily_key(_USER)] = 2_000_000
    usage[rate_limit._weekly_key(_USER)] = 5_000_000
    daily, weekly, tier = await rate_limit.get_global_rate_limits(_USER, 1, 1)

    with pytest.raises(rate_limit.RateLimitExceeded) as exc:
        await rate_limit.check_rate_limit(_USER, daily, weekly)

    assert exc.value.window == "daily"
    assert await rate_limit.get_remaining_usd_budget(_USER, daily, weekly) == 0.0
    status = await rate_limit.get_usage_status(_USER, daily, weekly, tier=tier)
    assert status.weekly.used == 5_000_000
    assert status.weekly.limit == 10_000_000
    assert status.weekly.resets_at == _SUNDAY + timedelta(minutes=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", [5_000_000, 20_000_000])
async def test_lifetime_trial_budget_remains_authoritative_without_redis(
    trial: TrialState, usage: dict[str, int], mocker: MockerFixture, spent: int
):
    trial.cost_microdollars = spent
    mocker.patch.object(
        rate_limit,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("unavailable")),
    )
    daily, weekly, tier = await rate_limit.get_global_rate_limits(_USER, 1, 1)

    if spent < 20_000_000:
        await rate_limit.check_rate_limit(_USER, daily, weekly)
    else:
        with pytest.raises(rate_limit.RateLimitExceeded) as exc:
            await rate_limit.check_rate_limit(_USER, daily, weekly)
        assert exc.value.window == "trial"

    assert (
        await rate_limit.get_remaining_usd_budget(_USER, daily, weekly)
        == (20_000_000 - spent) / 1_000_000
    )
    status = await rate_limit.get_usage_status(_USER, daily, weekly, tier=tier)
    assert status.weekly.used == spent
    assert status.weekly.resets_at == trial.ends_at
