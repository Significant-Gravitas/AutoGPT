from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import rate_limit, usage_activation
from backend.data.pro_activation import UsageActivationState
from backend.data.subscription_trial import TrialState

USER = "trial-state-user"


@pytest.fixture
def state_db(mocker):
    db = MagicMock()
    db.get_usage_activation_state = AsyncMock(
        return_value=UsageActivationState(user_id=USER, tier="TRIAL", ready=True)
    )
    mocker.patch.object(usage_activation, "pro_activation_db", return_value=db)
    return db


@pytest.fixture
def trial_db(mocker):
    trial = MagicMock(spec=TrialState)
    trial.active = False
    trial.cost_microdollars = 0
    trial.offer = MagicMock(total_cost_limit=500)
    trial.ends_at = None
    db = MagicMock()
    db.get_subscription_trial = AsyncMock(return_value=trial)
    mocker.patch.object(rate_limit, "credit_db", return_value=db)
    return db


@pytest.fixture
def limits(mocker, state_db, trial_db):
    mocker.patch.object(
        rate_limit, "_fetch_cost_limits_flag", AsyncMock(return_value=None)
    )
    mocker.patch.object(
        rate_limit,
        "get_tier_multipliers",
        AsyncMock(return_value={"NO_TIER": 0.0}),
    )
    mocker.patch.object(rate_limit, "is_feature_enabled", AsyncMock(return_value=True))
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    mocker.patch.object(rate_limit, "get_redis_async", AsyncMock(return_value=redis))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_trial", [False, True])
async def test_inactive_trial_usage_remains_readable(limits, trial_db, missing_trial):
    if missing_trial:
        trial_db.get_subscription_trial.return_value = None
    daily, weekly, tier = await rate_limit.get_global_rate_limits(USER, 100, 500)

    status = await rate_limit.get_usage_status(USER, daily, weekly, tier=tier)

    assert status.tier == rate_limit.SubscriptionTier.NO_TIER
    assert status.daily.limit == status.weekly.limit == 0
    public = rate_limit.CoPilotUsagePublic.from_status(status)
    assert public.daily is not None and public.daily.percent_used == 100
    assert public.weekly is not None and public.weekly.percent_used == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_trial", [False, True])
async def test_inactive_trial_remaining_budget_is_zero(limits, trial_db, missing_trial):
    if missing_trial:
        trial_db.get_subscription_trial.return_value = None
    daily, weekly, tier = await rate_limit.get_global_rate_limits(USER, 100, 500)

    remaining = await rate_limit.get_remaining_usd_budget(
        USER, daily, weekly, expected_tier=tier
    )

    assert remaining == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_trial", [False, True])
async def test_inactive_trial_still_rejects_spending(limits, trial_db, missing_trial):
    if missing_trial:
        trial_db.get_subscription_trial.return_value = None
    daily, weekly, tier = await rate_limit.get_global_rate_limits(USER, 100, 500)

    with pytest.raises(rate_limit.RateLimitExceeded) as error:
        await rate_limit.check_rate_limit(USER, daily, weekly, expected_tier=tier)

    assert error.value.window == "trial"


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["status", "budget", "limit"])
@pytest.mark.parametrize(
    "stored_tier,active,expected_tier",
    [
        ("PRO", False, "TRIAL"),
        ("TRIAL", True, "NO_TIER"),
        ("TRIAL", False, "TRIAL"),
    ],
)
async def test_tier_transition_still_fails_closed(
    limits, state_db, trial_db, stored_tier, active, expected_tier, consumer
):
    state_db.get_usage_activation_state.return_value = UsageActivationState(
        user_id=USER, tier=stored_tier, ready=True
    )
    trial_db.get_subscription_trial.return_value.active = active
    tier = rate_limit.SubscriptionTier(expected_tier)

    with pytest.raises(rate_limit.RateLimitUnavailable):
        await _read_usage(consumer, tier)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["status", "budget", "limit"])
async def test_unready_snapshot_still_fails_closed(limits, state_db, consumer):
    state_db.get_usage_activation_state.return_value = UsageActivationState(
        user_id=USER, tier="TRIAL", ready=False
    )

    with pytest.raises(rate_limit.RateLimitUnavailable):
        await _read_usage(consumer, rate_limit.SubscriptionTier.NO_TIER)


async def _read_usage(consumer: str, tier: rate_limit.SubscriptionTier) -> None:
    if consumer == "status":
        await rate_limit.get_usage_status(USER, 100, 500, tier=tier)
    elif consumer == "budget":
        await rate_limit.get_remaining_usd_budget(USER, 100, 500, expected_tier=tier)
    else:
        await rate_limit.check_rate_limit(USER, 100, 500, expected_tier=tier)
