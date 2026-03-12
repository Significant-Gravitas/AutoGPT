"""Unit tests for CoPilot rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from .rate_limit import (
    CoPilotUsageStatus,
    RateLimitExceeded,
    check_rate_limit,
    get_usage_status,
    record_token_usage,
)

_USER = "test-user-rl"
_SESSION = "test-session-rl"


# ---------------------------------------------------------------------------
# RateLimitExceeded
# ---------------------------------------------------------------------------


class TestRateLimitExceeded:
    def test_message_contains_window_name(self):
        exc = RateLimitExceeded("session", datetime.now(UTC) + timedelta(hours=1))
        assert "session" in str(exc)

    def test_message_contains_reset_time(self):
        exc = RateLimitExceeded(
            "weekly", datetime.now(UTC) + timedelta(hours=2, minutes=30)
        )
        msg = str(exc)
        # Allow for slight timing drift (29m or 30m)
        assert "2h " in msg
        assert "Resets in" in msg

    def test_message_minutes_only_when_under_one_hour(self):
        exc = RateLimitExceeded("session", datetime.now(UTC) + timedelta(minutes=15))
        msg = str(exc)
        assert "Resets in" in msg
        # Should not have "0h"
        assert "0h" not in msg


# ---------------------------------------------------------------------------
# get_usage_status
# ---------------------------------------------------------------------------


class TestGetUsageStatus:
    @pytest.mark.asyncio
    async def test_returns_redis_values(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500", "2000"])
        mock_redis.ttl = AsyncMock(return_value=7200)  # 2 hours remaining

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, _SESSION, session_token_limit=10000, weekly_token_limit=50000
            )

        assert isinstance(status, CoPilotUsageStatus)
        assert status.session.used == 500
        assert status.session.limit == 10000
        assert status.weekly.used == 2000
        assert status.weekly.limit == 50000

    @pytest.mark.asyncio
    async def test_returns_zeros_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            status = await get_usage_status(
                _USER, _SESSION, session_token_limit=10000, weekly_token_limit=50000
            )

        assert status.session.used == 0
        assert status.weekly.used == 0


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
                _USER, _SESSION, session_token_limit=10000, weekly_token_limit=50000
            )

    @pytest.mark.asyncio
    async def test_raises_when_session_limit_exceeded(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["10000", "200"])
        mock_redis.ttl = AsyncMock(return_value=3600)  # 1 hour remaining

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(
                    _USER, _SESSION, session_token_limit=10000, weekly_token_limit=50000
                )
            assert exc_info.value.window == "session"

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
                    _USER, _SESSION, session_token_limit=10000, weekly_token_limit=50000
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
                _USER, _SESSION, session_token_limit=10000, weekly_token_limit=50000
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
            await check_rate_limit(
                _USER, _SESSION, session_token_limit=0, weekly_token_limit=0
            )


# ---------------------------------------------------------------------------
# record_token_usage
# ---------------------------------------------------------------------------


class TestRecordTokenUsage:
    @pytest.mark.asyncio
    async def test_increments_redis_counters(self):
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(return_value=[])
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kwargs: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_token_usage(
                _USER, _SESSION, prompt_tokens=100, completion_tokens=50
            )

        # Should call incrby twice (session + weekly) with total=150
        incrby_calls = mock_pipe.incrby.call_args_list
        assert len(incrby_calls) == 2
        assert incrby_calls[0].args[1] == 150  # session
        assert incrby_calls[1].args[1] == 150  # weekly

    @pytest.mark.asyncio
    async def test_skips_when_zero_tokens(self):
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_token_usage(
                _USER, _SESSION, prompt_tokens=0, completion_tokens=0
            )

        # Should not call pipeline at all
        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_redis_failure_gracefully(self):
        """Should not raise when Redis is unavailable."""
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            # Should not raise
            await record_token_usage(
                _USER, _SESSION, prompt_tokens=100, completion_tokens=50
            )
