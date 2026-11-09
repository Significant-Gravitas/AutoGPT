import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.enums import SubscriptionTier
from redis.cluster import key_slot
from redis.exceptions import RedisClusterException, RedisError

from backend.api.features.onboarding_dump import rate_limit
from backend.util.settings import BehaveAs


@pytest.fixture
def admission(mocker, set_platform_payment):
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.CLOUD)
    set_platform_payment(True)
    tier_lookup = mocker.patch.object(
        rate_limit,
        "get_user_subscription_tier",
        new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
    )
    redis = MagicMock()
    redis.eval = AsyncMock(return_value=0)
    connect = mocker.patch.object(
        rate_limit, "get_redis_async", new=AsyncMock(return_value=redis)
    )
    return set_platform_payment, tier_lookup, redis, connect


@pytest.mark.asyncio
async def test_admitted_personalization_uses_one_atomic_budget(admission):
    _, _, redis, _ = admission
    await rate_limit.enforce_personalization_budget("user-1")
    redis.eval.assert_awaited_once()
    args = redis.eval.await_args.args
    assert args[1] == 2
    assert key_slot(args[2].encode()) == key_slot(args[3].encode())
    assert "user-1" not in args[2]
    assert args[4:6] == ("10", "30")


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [1, 3600, 86400])
async def test_quota_rejection_exposes_retry_after(admission, retry_after):
    _, _, redis, _ = admission
    redis.eval.return_value = retry_after
    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_personalization_budget("user-1")
    assert caught.value.status_code == 429
    assert caught.value.headers == {"Retry-After": str(retry_after)}


@pytest.mark.asyncio
async def test_active_plan_does_not_use_budget(admission):
    _, tier_lookup, _, connect = admission
    tier_lookup.return_value = SubscriptionTier.PRO
    await rate_limit.enforce_personalization_budget("user-1")
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_self_host_does_not_read_tier_or_use_budget(admission, mocker):
    _, tier_lookup, _, connect = admission
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.LOCAL)
    await rate_limit.enforce_personalization_budget("user-1")
    tier_lookup.assert_not_awaited()
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_payments_do_not_use_budget(admission):
    set_payment, _, _, connect = admission
    set_payment(False)
    await rate_limit.enforce_personalization_budget("user-1")
    connect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RedisError("down"), RedisClusterException("down")])
async def test_redis_error_blocks_new_paid_work_with_retry(admission, error):
    _, _, redis, _ = admission
    redis.eval.side_effect = error
    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_personalization_budget("user-1")
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "30"}


@pytest.mark.asyncio
async def test_entitlement_error_cannot_bypass_budget(admission):
    _, tier_lookup, _, connect = admission
    tier_lookup.side_effect = RuntimeError("tier unavailable")
    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_personalization_budget("user-1")
    assert caught.value.status_code == 503
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_timeout_is_bounded(admission, mocker):
    _, _, _, connect = admission

    async def stalled():
        await asyncio.sleep(10)

    connect.side_effect = stalled
    mocker.patch.object(rate_limit, "REDIS_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(HTTPException) as caught:
        await asyncio.wait_for(
            rate_limit.enforce_personalization_budget("user-1"), timeout=1
        )
    assert caught.value.status_code == 503


def test_windows_are_per_user_and_roll_over_independently():
    now = datetime(2026, 10, 6, 12, 30, tzinfo=UTC)
    hour_later = datetime(2026, 10, 6, 13, 30, tzinfo=UTC)
    tomorrow = datetime(2026, 10, 7, 12, 30, tzinfo=UTC)
    first = rate_limit._budget_windows("user-1", now)
    other = rate_limit._budget_windows("user-2", now)
    later = rate_limit._budget_windows("user-1", hour_later)
    next_day = rate_limit._budget_windows("user-1", tomorrow)
    assert first[0] != other[0]
    assert first[1] != other[1]
    assert first[0] != later[0]
    assert first[1] == later[1]
    assert first[1] != next_day[1]
    assert first[2:] == (1800, 41400)


# ---------- the onboarding interview's turns ----------


@pytest.mark.asyncio
async def test_an_interview_turn_uses_its_own_atomic_budget(admission):
    _, _, redis, _ = admission
    await rate_limit.enforce_interview_turn_budget("user-1")
    redis.eval.assert_awaited_once()
    args = redis.eval.await_args.args
    assert key_slot(args[2].encode()) == key_slot(args[3].encode())
    assert "user-1" not in args[2]
    assert args[2].startswith("onboarding:interview:")
    assert args[4:6] == (
        str(rate_limit.INTERVIEW_HOURLY_TURNS),
        str(rate_limit.INTERVIEW_DAILY_TURNS),
    )


def test_interview_turns_never_spend_personalization_attempts():
    now = datetime(2026, 10, 6, 12, 30, tzinfo=UTC)
    personalization = rate_limit._budget_windows("user-1", now)
    interview = rate_limit._budget_windows("user-1", now, "interview")
    assert personalization[:2] != interview[:2]
    assert personalization[0].startswith("onboarding:personalization:")


def test_a_whole_interview_fits_in_an_hour_of_the_budget():
    from backend.copilot.onboarding_interview import INTERVIEW_TURN_CAP

    assert rate_limit.INTERVIEW_HOURLY_TURNS > INTERVIEW_TURN_CAP
    assert rate_limit.INTERVIEW_DAILY_TURNS >= rate_limit.INTERVIEW_HOURLY_TURNS


@pytest.mark.asyncio
async def test_a_spent_interview_budget_is_a_429_with_retry_after(admission):
    _, _, redis, _ = admission
    redis.eval.return_value = 600
    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_interview_turn_budget("user-1")
    assert caught.value.status_code == 429
    assert caught.value.headers == {"Retry-After": "600"}


@pytest.mark.asyncio
async def test_an_unreadable_interview_budget_is_a_503(admission):
    _, _, redis, _ = admission
    redis.eval.side_effect = RedisError("down")
    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_interview_turn_budget("user-1")
    assert caught.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", list(SubscriptionTier))
async def test_an_interview_uses_its_budget_on_every_tier(admission, tier):
    """One counter for everyone, whatever their plan."""
    _, tier_lookup, redis, _ = admission
    tier_lookup.return_value = tier
    await rate_limit.enforce_interview_turn_budget("user-1")
    redis.eval.assert_awaited_once()
    assert redis.eval.await_args.args[2].startswith("onboarding:interview:")
    tier_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_paying_person_s_personalization_still_uses_no_budget(admission):
    _, tier_lookup, _, connect = admission
    tier_lookup.return_value = SubscriptionTier.PRO
    await rate_limit.enforce_personalization_budget("user-1")
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_self_hosted_interview_is_never_counted(admission, mocker):
    _, tier_lookup, _, connect = admission
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.LOCAL)
    await rate_limit.enforce_interview_turn_budget("user-1")
    tier_lookup.assert_not_awaited()
    connect.assert_not_awaited()
