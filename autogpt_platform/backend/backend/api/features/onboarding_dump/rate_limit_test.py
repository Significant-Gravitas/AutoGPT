import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
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


# ---------- the onboarding voice interview's calls ----------


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", list(SubscriptionTier))
async def test_a_voice_call_uses_its_own_budget_on_every_tier(admission, tier):
    _, tier_lookup, redis, _ = admission
    tier_lookup.return_value = tier
    await rate_limit.enforce_voice_call_budget("user-1")
    redis.eval.assert_awaited_once()
    args = redis.eval.await_args.args
    assert key_slot(args[2].encode()) == key_slot(args[3].encode())
    assert "user-1" not in args[2]
    assert args[2].startswith("onboarding:voice_interview:")
    assert args[4:6] == (
        str(rate_limit.VOICE_HOURLY_CALLS),
        str(rate_limit.VOICE_DAILY_CALLS),
    )
    tier_lookup.assert_not_awaited()


def test_voice_calls_never_spend_typed_interview_turns():
    now = datetime(2026, 11, 9, 12, 30, tzinfo=UTC)
    typed = rate_limit._budget_windows("user-1", now, "interview")
    voice = rate_limit._budget_windows("user-1", now, "voice_interview")
    assert typed[:2] != voice[:2]


@pytest.mark.asyncio
async def test_a_spent_voice_call_budget_is_a_429_with_retry_after(admission):
    _, _, redis, _ = admission
    redis.eval.return_value = 1200
    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_voice_call_budget("user-1")
    assert caught.value.status_code == 429
    assert caught.value.headers == {"Retry-After": "1200"}


@pytest.mark.asyncio
async def test_a_self_hosted_voice_call_is_never_counted(admission, mocker):
    _, _, _, connect = admission
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.LOCAL)
    await rate_limit.enforce_voice_call_budget("user-1")
    connect.assert_not_awaited()


@pytest.fixture
async def voice_counters(admission, mocker):
    """The voice call allowance's counters in a Redis of their own."""
    redis = FakeRedis(server=FakeServer(), decode_responses=True)
    mocker.patch.object(
        rate_limit, "get_redis_async", new=AsyncMock(return_value=redis)
    )
    yield redis
    await redis.aclose()


@pytest.mark.asyncio
async def test_reading_the_voice_call_allowance_counts_no_call(voice_counters):
    for _ in range(rate_limit.VOICE_HOURLY_CALLS + 1):
        await rate_limit.enforce_voice_call_budget("user-1")

    assert await voice_counters.keys("onboarding:voice_interview:*") == []


@pytest.mark.asyncio
async def test_calls_the_relay_counts_spend_the_allowance_the_start_reads(
    voice_counters,
):
    for _ in range(rate_limit.VOICE_HOURLY_CALLS):
        await rate_limit.count_voice_call(voice_counters, "user-1")

    with pytest.raises(HTTPException) as caught:
        await rate_limit.enforce_voice_call_budget("user-1")

    assert caught.value.status_code == 429
    assert 0 < int(caught.value.headers["Retry-After"]) <= 3600
    # Another person's allowance is their own.
    await rate_limit.enforce_voice_call_budget("user-2")


@pytest.mark.asyncio
async def test_the_relay_counts_a_call_with_its_windows_ttls(voice_counters):
    await rate_limit.count_voice_call(voice_counters, "user-1")

    keys = sorted(await voice_counters.keys("onboarding:voice_interview:*"))
    assert len(keys) == 2
    for key in keys:
        assert await voice_counters.get(key) == "1"
        assert 0 < await voice_counters.ttl(key) <= 86_400


@pytest.mark.asyncio
async def test_a_self_hosted_call_is_never_counted_by_the_relay(voice_counters, mocker):
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.LOCAL)

    await rate_limit.count_voice_call(voice_counters, "user-1")

    assert await voice_counters.keys("onboarding:voice_interview:*") == []
