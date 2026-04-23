"""Unit tests for CoPilot rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from .rate_limit import (
    DEFAULT_TIER,
    TIER_MULTIPLIERS,
    CoPilotUsageStatus,
    RateLimitExceeded,
    SubscriptionTier,
    UsageWindow,
    _daily_key,
    _daily_reset_time,
    _weekly_key,
    _weekly_reset_time,
    acquire_reset_lock,
    check_rate_limit,
    get_daily_reset_count,
    get_global_rate_limits,
    get_usage_status,
    get_user_tier,
    increment_daily_reset_count,
    record_cost_usage,
    release_reset_lock,
    reset_daily_usage,
    reset_user_usage,
    set_user_tier,
)

_USER = "test-user-rl"


# ---------------------------------------------------------------------------
# RateLimitExceeded
# ---------------------------------------------------------------------------


class TestRateLimitExceeded:
    def test_message_contains_window_name(self):
        exc = RateLimitExceeded("daily", datetime.now(UTC) + timedelta(hours=1))
        assert "daily" in str(exc)

    def test_message_contains_reset_time(self):
        exc = RateLimitExceeded(
            "weekly", datetime.now(UTC) + timedelta(hours=2, minutes=30)
        )
        msg = str(exc)
        # Allow for slight timing drift (29m or 30m)
        assert "2h " in msg
        assert "Resets in" in msg

    def test_message_minutes_only_when_under_one_hour(self):
        exc = RateLimitExceeded("daily", datetime.now(UTC) + timedelta(minutes=15))
        msg = str(exc)
        assert "Resets in" in msg
        # Should not have "0h"
        assert "0h" not in msg

    def test_message_says_now_when_resets_at_is_in_the_past(self):
        """Negative delta (clock skew / stale TTL) should say 'now', not '-1h -30m'."""
        exc = RateLimitExceeded("daily", datetime.now(UTC) - timedelta(minutes=5))
        assert "Resets in now" in str(exc)


# ---------------------------------------------------------------------------
# get_usage_status
# ---------------------------------------------------------------------------


class TestGetUsageStatus:
    @pytest.mark.asyncio
    async def test_returns_redis_values(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500", "2000"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert isinstance(status, CoPilotUsageStatus)
        assert status.daily.used == 500
        assert status.daily.limit == 10000
        assert status.weekly.used == 2000
        assert status.weekly.limit == 50000

    @pytest.mark.asyncio
    async def test_returns_zeros_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert status.daily.used == 0
        assert status.weekly.used == 0

    @pytest.mark.asyncio
    async def test_partial_none_daily_counter(self):
        """Daily counter is None (new day), weekly has usage."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[None, "3000"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert status.daily.used == 0
        assert status.weekly.used == 3000

    @pytest.mark.asyncio
    async def test_partial_none_weekly_counter(self):
        """Weekly counter is None (start of week), daily has usage."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500", None])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert status.daily.used == 500
        assert status.weekly.used == 0

    @pytest.mark.asyncio
    async def test_resets_at_daily_is_next_midnight_utc(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["0", "0"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        now = datetime.now(UTC)
        # Daily reset should be within 24h
        assert status.daily.resets_at > now
        assert status.daily.resets_at <= now + timedelta(hours=24, seconds=5)


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_allows_when_under_limit(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100", "200"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            # Should not raise
            await check_rate_limit(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

    @pytest.mark.asyncio
    async def test_raises_when_daily_limit_exceeded(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["10000", "200"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )
            assert exc_info.value.window == "daily"

    @pytest.mark.asyncio
    async def test_raises_when_weekly_limit_exceeded(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100", "50000"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )
            assert exc_info.value.window == "weekly"

    @pytest.mark.asyncio
    async def test_allows_when_redis_unavailable(self):
        """Fail-open: allow requests when Redis is down."""
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            # Should not raise
            await check_rate_limit(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

    @pytest.mark.asyncio
    async def test_skips_check_when_limit_is_zero(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["999999", "999999"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            # Should not raise — limits of 0 mean unlimited
            await check_rate_limit(_USER, daily_cost_limit=0, weekly_cost_limit=0)


# ---------------------------------------------------------------------------
# record_cost_usage
# ---------------------------------------------------------------------------


class TestRecordCostUsage:
    @staticmethod
    def _make_pipeline_mock() -> MagicMock:
        """Create a pipeline mock with sync methods and async execute."""
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[])
        return pipe

    @pytest.mark.asyncio
    async def test_increments_redis_counters(self):
        mock_pipe = self._make_pipeline_mock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=123_456)

        # Should call incrby twice (daily + weekly) with the same cost
        incrby_calls = mock_pipe.incrby.call_args_list
        assert len(incrby_calls) == 2
        assert incrby_calls[0].args[1] == 123_456  # daily
        assert incrby_calls[1].args[1] == 123_456  # weekly

    @pytest.mark.asyncio
    async def test_skips_when_cost_is_zero(self):
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=0)

        # Should not call pipeline at all
        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_cost_is_negative(self):
        """Negative costs are clamped to zero and skip the pipeline."""
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=-10)

        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_sets_expire_on_both_keys(self):
        """Pipeline should call expire for both daily and weekly keys."""
        mock_pipe = self._make_pipeline_mock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=5_000)

        expire_calls = mock_pipe.expire.call_args_list
        assert len(expire_calls) == 2

        # Daily key TTL should be positive (seconds until next midnight)
        daily_ttl = expire_calls[0].args[1]
        assert daily_ttl >= 1

        # Weekly key TTL should be positive (seconds until next Monday)
        weekly_ttl = expire_calls[1].args[1]
        assert weekly_ttl >= 1

    @pytest.mark.asyncio
    async def test_handles_redis_failure_gracefully(self):
        """Should not raise when Redis is unavailable."""
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            # Should not raise
            await record_cost_usage(_USER, cost_microdollars=5_000)

    @pytest.mark.asyncio
    async def test_handles_redis_error_during_pipeline_execute(self):
        """Should not raise when pipeline.execute() fails with RedisError."""
        mock_pipe = self._make_pipeline_mock()
        mock_pipe.execute = AsyncMock(side_effect=RedisError("Pipeline failed"))
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            # Should not raise — fail-open
            await record_cost_usage(_USER, cost_microdollars=5_000)


# ---------------------------------------------------------------------------
# SubscriptionTier and tier multipliers
# ---------------------------------------------------------------------------


class TestSubscriptionTier:
    def test_tier_values(self):
        assert SubscriptionTier.FREE.value == "FREE"
        assert SubscriptionTier.PRO.value == "PRO"
        assert SubscriptionTier.BUSINESS.value == "BUSINESS"
        assert SubscriptionTier.ENTERPRISE.value == "ENTERPRISE"

    def test_tier_multipliers(self):
        assert TIER_MULTIPLIERS[SubscriptionTier.FREE] == 1
        assert TIER_MULTIPLIERS[SubscriptionTier.PRO] == 5
        assert TIER_MULTIPLIERS[SubscriptionTier.BUSINESS] == 20
        assert TIER_MULTIPLIERS[SubscriptionTier.ENTERPRISE] == 60

    def test_default_tier_is_free(self):
        assert DEFAULT_TIER == SubscriptionTier.FREE

    def test_usage_status_includes_tier(self):
        now = datetime.now(UTC)
        status = CoPilotUsageStatus(
            daily=UsageWindow(used=0, limit=100, resets_at=now + timedelta(hours=1)),
            weekly=UsageWindow(used=0, limit=500, resets_at=now + timedelta(days=1)),
        )
        assert status.tier == SubscriptionTier.FREE

    def test_usage_status_with_custom_tier(self):
        now = datetime.now(UTC)
        status = CoPilotUsageStatus(
            daily=UsageWindow(used=0, limit=100, resets_at=now + timedelta(hours=1)),
            weekly=UsageWindow(used=0, limit=500, resets_at=now + timedelta(days=1)),
            tier=SubscriptionTier.PRO,
        )
        assert status.tier == SubscriptionTier.PRO


# ---------------------------------------------------------------------------
# get_user_tier
# ---------------------------------------------------------------------------


class TestGetUserTier:
    @pytest.fixture(autouse=True)
    def _clear_tier_cache(self):
        """Clear the get_user_tier cache before each test."""
        get_user_tier.cache_clear()  # type: ignore[attr-defined]

    def _mock_user_db(
        self, subscription_tier: str | None = None, raises: Exception | None = None
    ):
        """Return a patched user_db() whose get_user_by_id behaves as specified."""
        mock_db = AsyncMock()
        if raises is not None:
            mock_db.get_user_by_id = AsyncMock(side_effect=raises)
        else:
            mock_user = MagicMock()
            mock_user.subscription_tier = subscription_tier
            mock_db.get_user_by_id = AsyncMock(return_value=mock_user)
        return mock_db

    @pytest.mark.asyncio
    async def test_returns_tier_from_db(self):
        """Should return the tier stored in the user record."""
        mock_db = self._mock_user_db(subscription_tier="PRO")
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_returns_default_when_user_not_found(self):
        """Should return DEFAULT_TIER when user is not in the DB."""
        mock_db = self._mock_user_db(raises=Exception("not found"))
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_returns_default_when_tier_is_none(self):
        """Should return DEFAULT_TIER when subscription_tier is None."""
        mock_db = self._mock_user_db(subscription_tier=None)
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_returns_default_on_db_error(self):
        """Should fall back to DEFAULT_TIER when DB raises."""
        mock_db = self._mock_user_db(raises=Exception("DB down"))
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_db_error_is_not_cached(self):
        """Transient DB errors should NOT cache the default tier.

        Regression test: a transient DB failure previously cached DEFAULT_TIER
        for 5 minutes, incorrectly downgrading higher-tier users until expiry.
        """
        failing_db = self._mock_user_db(raises=Exception("DB down"))
        with patch("backend.copilot.rate_limit.user_db", return_value=failing_db):
            tier1 = await get_user_tier(_USER)
        assert tier1 == DEFAULT_TIER

        # Now DB recovers and returns PRO
        ok_db = self._mock_user_db(subscription_tier="PRO")
        with patch("backend.copilot.rate_limit.user_db", return_value=ok_db):
            tier2 = await get_user_tier(_USER)

        # Should get PRO now — the error result was not cached
        assert tier2 == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_returns_default_on_invalid_tier_value(self):
        """Should fall back to DEFAULT_TIER when stored value is invalid."""
        mock_db = self._mock_user_db(subscription_tier="invalid-tier")
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_user_not_found_is_not_cached(self):
        """Non-existent user should NOT cache DEFAULT_TIER.

        Regression test: when ``get_user_tier`` is called before a user record
        exists, the DEFAULT_TIER fallback must not be cached.  Otherwise, a
        newly created user with a higher tier (e.g. PRO) would receive the
        stale cached FREE tier for up to 5 minutes.
        """
        # First call: user does not exist yet
        missing_db = self._mock_user_db(raises=Exception("not found"))
        with patch("backend.copilot.rate_limit.user_db", return_value=missing_db):
            tier1 = await get_user_tier(_USER)
        assert tier1 == DEFAULT_TIER

        # Second call: user now exists with PRO tier
        ok_db = self._mock_user_db(subscription_tier="PRO")
        with patch("backend.copilot.rate_limit.user_db", return_value=ok_db):
            tier2 = await get_user_tier(_USER)

        # Should get PRO — the not-found result was not cached
        assert tier2 == SubscriptionTier.PRO


# ---------------------------------------------------------------------------
# set_user_tier
# ---------------------------------------------------------------------------


class TestSetUserTier:
    @pytest.fixture(autouse=True)
    def _clear_tier_cache(self):
        """Clear the get_user_tier cache before each test."""
        get_user_tier.cache_clear()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_updates_db_and_invalidates_cache(self):
        """set_user_tier should persist to DB and invalidate the tier cache."""
        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(return_value=None)

        with patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=mock_prisma,
        ):
            await set_user_tier(_USER, SubscriptionTier.PRO)

        mock_prisma.update.assert_awaited_once_with(
            where={"id": _USER},
            data={"subscriptionTier": "PRO"},
        )

    @pytest.mark.asyncio
    async def test_record_not_found_propagates(self):
        """RecordNotFoundError from Prisma should propagate to callers."""
        import prisma.errors

        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(
            side_effect=prisma.errors.RecordNotFoundError(
                {"error": "Record not found"}
            ),
        )

        with patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=mock_prisma,
        ):
            with pytest.raises(prisma.errors.RecordNotFoundError):
                await set_user_tier(_USER, SubscriptionTier.ENTERPRISE)

    @pytest.mark.asyncio
    async def test_cache_invalidated_after_set(self):
        """After set_user_tier, get_user_tier should query DB again (not cache)."""
        # First, populate the cache with BUSINESS via user_db() mock
        mock_db_biz = AsyncMock()
        mock_user_biz = MagicMock()
        mock_user_biz.subscription_tier = "BUSINESS"
        mock_db_biz.get_user_by_id = AsyncMock(return_value=mock_user_biz)

        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db_biz):
            tier_before = await get_user_tier(_USER)
        assert tier_before == SubscriptionTier.BUSINESS

        # Now set tier to ENTERPRISE via PrismaUser.prisma (set_user_tier still
        # uses Prisma directly since it's only called from admin API where Prisma
        # is connected).
        mock_prisma_set = AsyncMock()
        mock_prisma_set.update = AsyncMock(return_value=None)

        with patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=mock_prisma_set,
        ):
            await set_user_tier(_USER, SubscriptionTier.ENTERPRISE)

        # Now get_user_tier should hit DB again (cache was invalidated)
        mock_db_ent = AsyncMock()
        mock_user_ent = MagicMock()
        mock_user_ent.subscription_tier = "ENTERPRISE"
        mock_db_ent.get_user_by_id = AsyncMock(return_value=mock_user_ent)

        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db_ent):
            tier_after = await get_user_tier(_USER)

        assert tier_after == SubscriptionTier.ENTERPRISE

    @pytest.mark.asyncio
    async def test_drift_check_swallows_launchdarkly_failure(self):
        """LaunchDarkly price-id lookup failures inside the drift check must
        never bubble up and 500 the admin tier write — the DB update is
        already committed by the time we check drift."""
        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(return_value=None)

        mock_user = MagicMock()
        mock_user.stripe_customer_id = "cus_abc"

        mock_sub = MagicMock()
        mock_sub.id = "sub_abc"
        mock_sub["items"].data = [MagicMock(price=MagicMock(id="price_mismatch"))]

        with (
            patch(
                "backend.copilot.rate_limit.PrismaUser.prisma",
                return_value=mock_prisma,
            ),
            patch(
                "backend.copilot.rate_limit.get_user_by_id",
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
            patch(
                "backend.data.credit._get_active_subscription",
                new_callable=AsyncMock,
                return_value=mock_sub,
            ),
            patch(
                "backend.data.credit.get_subscription_price_id",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LD SDK not initialized"),
            ),
        ):
            # Must NOT raise — drift check is best-effort diagnostic only.
            await set_user_tier(_USER, SubscriptionTier.PRO)

        mock_prisma.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drift_check_timeout_is_bounded(self):
        """A Stripe call that stalls on the 80s SDK default must not block the
        admin tier write — set_user_tier wraps the drift check in a 5s timeout
        and logs + returns on TimeoutError."""
        import asyncio as _asyncio

        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(return_value=None)

        async def _never_returns(_user_id: str, _tier):
            await _asyncio.sleep(60)

        with (
            patch(
                "backend.copilot.rate_limit.PrismaUser.prisma",
                return_value=mock_prisma,
            ),
            patch(
                "backend.copilot.rate_limit._warn_if_stripe_subscription_drifts",
                side_effect=_never_returns,
            ),
            patch(
                "backend.copilot.rate_limit.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=_asyncio.TimeoutError,
            ),
        ):
            await set_user_tier(_USER, SubscriptionTier.PRO)

        # Set_user_tier still completed — the drift timeout did not propagate.
        mock_prisma.update.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_global_rate_limits with tiers
# ---------------------------------------------------------------------------


class TestGetGlobalRateLimitsWithTiers:
    @staticmethod
    def _ld_side_effect(daily: int, weekly: int):
        """Return an async side_effect that dispatches by flag_key."""

        async def _side_effect(flag_key: str, _uid: str, default: int) -> int:
            if "daily" in flag_key.lower():
                return daily
            if "weekly" in flag_key.lower():
                return weekly
            return default

        return _side_effect

    @pytest.mark.asyncio
    async def test_free_tier_no_multiplier(self):
        """Free tier should not change limits."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.FREE,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 2_500_000
        assert weekly == 12_500_000
        assert tier == SubscriptionTier.FREE

    @pytest.mark.asyncio
    async def test_pro_tier_5x_multiplier(self):
        """Pro tier should multiply limits by 5."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 12_500_000
        assert weekly == 62_500_000
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_business_tier_20x_multiplier(self):
        """Business tier should multiply limits by 20."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 50_000_000
        assert weekly == 250_000_000
        assert tier == SubscriptionTier.BUSINESS

    @pytest.mark.asyncio
    async def test_enterprise_tier_60x_multiplier(self):
        """Enterprise tier should multiply limits by 60."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.ENTERPRISE,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 150_000_000
        assert weekly == 750_000_000
        assert tier == SubscriptionTier.ENTERPRISE


# ---------------------------------------------------------------------------
# End-to-end: tier limits are respected by check_rate_limit
# ---------------------------------------------------------------------------


class TestTierLimitsRespected:
    """Verify that tier-adjusted limits from get_global_rate_limits flow
    correctly into check_rate_limit, so higher tiers allow more usage and
    lower tiers are blocked when they would exceed their allocation."""

    _BASE_DAILY = 2_500_000
    _BASE_WEEKLY = 12_500_000

    @staticmethod
    def _ld_side_effect(daily: int, weekly: int):

        async def _side_effect(flag_key: str, _uid: str, default: int) -> int:
            if "daily" in flag_key.lower():
                return daily
            if "weekly" in flag_key.lower():
                return weekly
            return default

        return _side_effect

    @pytest.mark.asyncio
    async def test_pro_user_allowed_above_free_limit(self):
        """A PRO user with usage above the FREE limit should be allowed."""
        # Usage: 3M tokens (above FREE limit of 2.5M, below PRO limit of 12.5M)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["3000000", "3000000"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            # PRO: 5x multiplier
            assert daily == 12_500_000
            assert tier == SubscriptionTier.PRO
            # Should NOT raise — 3M < 12.5M
            await check_rate_limit(
                _USER, daily_cost_limit=daily, weekly_cost_limit=weekly
            )

    @pytest.mark.asyncio
    async def test_free_user_blocked_at_free_limit(self):
        """A FREE user at or above the base limit should be blocked."""
        # Usage: 2.5M tokens (at FREE limit of 2.5M)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["2500000", "2500000"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.FREE,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            # FREE: 1x multiplier
            assert daily == 2_500_000
            assert tier == SubscriptionTier.FREE
            # Should raise — 2.5M >= 2.5M
            with pytest.raises(RateLimitExceeded):
                await check_rate_limit(
                    _USER, daily_cost_limit=daily, weekly_cost_limit=weekly
                )

    @pytest.mark.asyncio
    async def test_enterprise_user_has_highest_headroom(self):
        """An ENTERPRISE user should have 60x the base limit."""
        # Usage: 100M tokens (huge, but below ENTERPRISE daily of 150M)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100000000", "100000000"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.ENTERPRISE,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert daily == 150_000_000
            assert tier == SubscriptionTier.ENTERPRISE
            # Should NOT raise — 100M < 150M
            await check_rate_limit(
                _USER, daily_cost_limit=daily, weekly_cost_limit=weekly
            )


# ---------------------------------------------------------------------------
# reset_daily_usage
# ---------------------------------------------------------------------------


class TestResetDailyUsage:
    @staticmethod
    def _make_pipeline_mock(decrby_result: int = 0) -> MagicMock:
        """Create a pipeline mock that returns [delete_result, decrby_result]."""
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[1, decrby_result])
        return pipe

    @pytest.mark.asyncio
    async def test_deletes_daily_key(self):
        mock_pipe = self._make_pipeline_mock(decrby_result=0)
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            result = await reset_daily_usage(_USER, daily_cost_limit=10000)

        assert result is True
        mock_pipe.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reduces_weekly_usage_via_decrby(self):
        """Weekly counter should be reduced via DECRBY in the pipeline."""
        mock_pipe = self._make_pipeline_mock(decrby_result=35000)
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await reset_daily_usage(_USER, daily_cost_limit=10000)

        mock_pipe.decrby.assert_called_once()
        mock_redis.set.assert_not_called()  # 35000 > 0, no clamp needed

    @pytest.mark.asyncio
    async def test_clamps_negative_weekly_to_zero(self):
        """If DECRBY goes negative, SET to 0 (outside the pipeline)."""
        mock_pipe = self._make_pipeline_mock(decrby_result=-5000)
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await reset_daily_usage(_USER, daily_cost_limit=10000)

        mock_pipe.decrby.assert_called_once()
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_weekly_reduction_when_daily_limit_zero(self):
        """When daily_cost_limit is 0, weekly counter should not be touched."""
        mock_pipe = self._make_pipeline_mock()
        mock_pipe.execute = AsyncMock(return_value=[1])  # only delete result
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await reset_daily_usage(_USER, daily_cost_limit=0)

        mock_pipe.delete.assert_called_once()
        mock_pipe.decrby.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            result = await reset_daily_usage(_USER, daily_cost_limit=10000)

        assert result is False


# ---------------------------------------------------------------------------
# Tier-limit enforcement (integration-style)
# ---------------------------------------------------------------------------


class TestTierLimitsEnforced:
    """Verify that tier-multiplied limits are actually respected by
    ``check_rate_limit`` — i.e. that usage within the tier allowance passes
    and usage at/above the tier allowance is rejected."""

    _BASE_DAILY = 1_000_000
    _BASE_WEEKLY = 5_000_000

    @staticmethod
    def _ld_side_effect(daily: int, weekly: int):
        """Mock LD flag lookup returning the given raw limits."""

        async def _side_effect(flag_key: str, _uid: str, default: int) -> int:
            if "daily" in flag_key.lower():
                return daily
            if "weekly" in flag_key.lower():
                return weekly
            return default

        return _side_effect

    @pytest.mark.asyncio
    async def test_pro_within_limit_allowed(self):
        """Usage under PRO daily limit should not raise."""
        pro_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO]
        mock_redis = AsyncMock()
        # Simulate usage just under the PRO daily limit
        mock_redis.get = AsyncMock(side_effect=[str(pro_daily - 1), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.PRO
            assert daily == pro_daily
            # Should not raise — usage is under the limit
            await check_rate_limit(_USER, daily, weekly)

    @pytest.mark.asyncio
    async def test_pro_at_limit_rejected(self):
        """Usage at exactly the PRO daily limit should raise."""
        pro_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO]
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(pro_daily), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily, weekly)
            assert exc_info.value.window == "daily"

    @pytest.mark.asyncio
    async def test_business_higher_limit_allows_pro_overflow(self):
        """Usage exceeding PRO but under BUSINESS should pass for BUSINESS."""
        pro_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO]
        biz_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.BUSINESS]
        # Usage between PRO and BUSINESS limits
        usage = pro_daily + 1_000_000
        assert usage < biz_daily, "test sanity: usage must be under BUSINESS limit"

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(usage), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.BUSINESS
            assert daily == biz_daily
            # Should not raise — BUSINESS tier can handle this
            await check_rate_limit(_USER, daily, weekly)

    @pytest.mark.asyncio
    async def test_weekly_limit_enforced_for_tier(self):
        """Weekly limit should also be tier-multiplied and enforced."""
        pro_weekly = self._BASE_WEEKLY * TIER_MULTIPLIERS[SubscriptionTier.PRO]
        mock_redis = AsyncMock()
        # Daily usage fine, weekly at limit
        mock_redis.get = AsyncMock(side_effect=["0", str(pro_weekly)])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily, weekly)
            assert exc_info.value.window == "weekly"

    @pytest.mark.asyncio
    async def test_free_tier_base_limit_enforced(self):
        """Free tier (1x multiplier) should enforce the base limit exactly."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(self._BASE_DAILY), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.FREE,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert daily == self._BASE_DAILY  # 1x multiplier
            with pytest.raises(RateLimitExceeded):
                await check_rate_limit(_USER, daily, weekly)

    @pytest.mark.asyncio
    async def test_free_tier_cannot_bypass_pro_limit(self):
        """A FREE-tier user whose usage is within PRO limits but over FREE
        limits must still be rejected.

        Negative test: ensures the tier multiplier is applied *before* the
        rate-limit check, so a lower-tier user cannot 'bypass' limits that
        would be acceptable for a higher tier.
        """
        free_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.FREE]
        pro_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO]
        # Usage above FREE limit but below PRO limit
        usage = free_daily + 500_000
        assert usage < pro_daily, "test sanity: usage must be under PRO limit"

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(usage), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.FREE,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.FREE
            assert daily == free_daily  # 1x, not 5x
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily, weekly)
            assert exc_info.value.window == "daily"

    @pytest.mark.asyncio
    async def test_tier_change_updates_effective_limits(self):
        """After upgrading from FREE to BUSINESS, the effective limits must
        increase accordingly.

        Verifies that the tier multiplier is correctly applied after a tier
        change, and that usage that was over the FREE limit is within the new
        BUSINESS limit.
        """
        free_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.FREE]
        biz_daily = self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.BUSINESS]
        # Usage above FREE limit but below BUSINESS limit
        usage = free_daily + 500_000
        assert usage < biz_daily, "test sanity: usage must be under BUSINESS limit"

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(usage), "0"])

        # Simulate the user having been upgraded to BUSINESS
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
            patch(
                "backend.util.feature_flag.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.BUSINESS
            assert daily == biz_daily  # 20x
            # Should NOT raise — usage is within the BUSINESS tier allowance
            await check_rate_limit(_USER, daily, weekly)


# ---------------------------------------------------------------------------
# Private key/reset helpers
# ---------------------------------------------------------------------------


class TestKeyHelpers:
    def test_daily_key_format(self):
        now = datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)
        key = _daily_key("user-1", now=now)
        assert "daily" in key
        assert "user-1" in key
        assert "2026-04-03" in key

    def test_daily_key_defaults_to_now(self):
        key = _daily_key("user-1")
        assert "daily" in key
        assert "user-1" in key

    def test_weekly_key_format(self):
        now = datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)
        key = _weekly_key("user-1", now=now)
        assert "weekly" in key
        assert "user-1" in key
        assert "2026-W" in key

    def test_weekly_key_defaults_to_now(self):
        key = _weekly_key("user-1")
        assert "weekly" in key

    def test_daily_reset_time_is_next_midnight(self):
        now = datetime(2026, 4, 3, 15, 30, 0, tzinfo=UTC)
        reset = _daily_reset_time(now=now)
        assert reset == datetime(2026, 4, 4, 0, 0, 0, tzinfo=UTC)

    def test_daily_reset_time_defaults_to_now(self):
        reset = _daily_reset_time()
        assert reset.hour == 0
        assert reset.minute == 0

    def test_weekly_reset_time_is_next_monday(self):
        # 2026-04-03 is a Friday
        now = datetime(2026, 4, 3, 15, 30, 0, tzinfo=UTC)
        reset = _weekly_reset_time(now=now)
        assert reset.weekday() == 0  # Monday
        assert reset == datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)

    def test_weekly_reset_time_defaults_to_now(self):
        reset = _weekly_reset_time()
        assert reset.weekday() == 0  # Monday


# ---------------------------------------------------------------------------
# acquire_reset_lock / release_reset_lock
# ---------------------------------------------------------------------------


class TestResetLock:
    @pytest.mark.asyncio
    async def test_acquire_lock_success(self):
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            result = await acquire_reset_lock("user-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_already_held(self):
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=False)
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            result = await acquire_reset_lock("user-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            result = await acquire_reset_lock("user-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_lock_success(self):
        mock_redis = AsyncMock()
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await release_reset_lock("user-1")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_lock_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            # Should not raise
            await release_reset_lock("user-1")


# ---------------------------------------------------------------------------
# get_daily_reset_count / increment_daily_reset_count
# ---------------------------------------------------------------------------


class TestDailyResetCount:
    @pytest.mark.asyncio
    async def test_get_count_returns_value(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="3")
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            count = await get_daily_reset_count("user-1")
        assert count == 3

    @pytest.mark.asyncio
    async def test_get_count_returns_zero_when_no_key(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            count = await get_daily_reset_count("user-1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_count_returns_none_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            count = await get_daily_reset_count("user-1")
        assert count is None

    @pytest.mark.asyncio
    async def test_increment_count(self):
        mock_pipe = MagicMock()
        mock_pipe.incr = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)

        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await increment_daily_reset_count("user-1")
        mock_pipe.incr.assert_called_once()
        mock_pipe.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_increment_count_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            # Should not raise
            await increment_daily_reset_count("user-1")


# ---------------------------------------------------------------------------
# reset_user_usage
# ---------------------------------------------------------------------------


class TestResetUserUsage:
    @pytest.mark.asyncio
    async def test_resets_daily_key(self):
        mock_redis = AsyncMock()
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await reset_user_usage("user-1")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_resets_daily_and_weekly(self):
        mock_redis = AsyncMock()
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await reset_user_usage("user-1", reset_weekly=True)
        args = mock_redis.delete.call_args[0]
        assert len(args) == 2  # both daily and weekly keys

    @pytest.mark.asyncio
    async def test_raises_on_redis_failure(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            with pytest.raises(RedisError):
                await reset_user_usage("user-1")
