"""Unit tests for CoPilot rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from .rate_limit import (
    CoPilotUsageStatus,
    RateLimitExceeded,
    check_rate_limit,
    get_usage_status,
    record_token_usage,
    reset_daily_usage,
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                    _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                    _USER, daily_token_limit=10000, weekly_token_limit=50000
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
                _USER, daily_token_limit=10000, weekly_token_limit=50000
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
            await check_rate_limit(_USER, daily_token_limit=0, weekly_token_limit=0)


# ---------------------------------------------------------------------------
# record_token_usage
# ---------------------------------------------------------------------------


class TestRecordTokenUsage:
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
            await record_token_usage(_USER, prompt_tokens=100, completion_tokens=50)

        # Should call incrby twice (daily + weekly) with total=150
        incrby_calls = mock_pipe.incrby.call_args_list
        assert len(incrby_calls) == 2
        assert incrby_calls[0].args[1] == 150  # daily
        assert incrby_calls[1].args[1] == 150  # weekly

    @pytest.mark.asyncio
    async def test_skips_when_zero_tokens(self):
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_token_usage(_USER, prompt_tokens=0, completion_tokens=0)

        # Should not call pipeline at all
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
            await record_token_usage(_USER, prompt_tokens=100, completion_tokens=50)

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
            await record_token_usage(_USER, prompt_tokens=100, completion_tokens=50)

    @pytest.mark.asyncio
    async def test_cost_weighted_counting(self):
        """Cached tokens should be weighted: cache_read=10%, cache_create=25%."""
        mock_pipe = self._make_pipeline_mock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_token_usage(
                _USER,
                prompt_tokens=100,  # uncached → 100
                completion_tokens=50,  # output → 50
                cache_read_tokens=10000,  # 10% → 1000
                cache_creation_tokens=400,  # 25% → 100
            )

        # Expected weighted total: 100 + 1000 + 100 + 50 = 1250
        incrby_calls = mock_pipe.incrby.call_args_list
        assert len(incrby_calls) == 2
        assert incrby_calls[0].args[1] == 1250  # daily
        assert incrby_calls[1].args[1] == 1250  # weekly

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
            await record_token_usage(_USER, prompt_tokens=100, completion_tokens=50)


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
            result = await reset_daily_usage(_USER, daily_token_limit=10000)

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
            await reset_daily_usage(_USER, daily_token_limit=10000)

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
            await reset_daily_usage(_USER, daily_token_limit=10000)

        mock_pipe.decrby.assert_called_once()
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_weekly_reduction_when_daily_limit_zero(self):
        """When daily_token_limit is 0, weekly counter should not be touched."""
        mock_pipe = self._make_pipeline_mock()
        mock_pipe.execute = AsyncMock(return_value=[1])  # only delete result
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await reset_daily_usage(_USER, daily_token_limit=0)

        mock_pipe.delete.assert_called_once()
        mock_pipe.decrby.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            result = await reset_daily_usage(_USER, daily_token_limit=10000)

        assert result is False
