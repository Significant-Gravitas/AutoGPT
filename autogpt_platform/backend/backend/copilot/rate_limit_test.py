"""Unit tests for CoPilot rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from .rate_limit import (
    _SESSION_TTL_SECONDS,
    CoPilotUsageStatus,
    RateLimitExceeded,
    _session_reset_from_ttl,
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

    def test_message_says_now_when_resets_at_is_in_the_past(self):
        """Negative delta (clock skew / stale TTL) should say 'now', not '-1h -30m'."""
        exc = RateLimitExceeded("session", datetime.now(UTC) - timedelta(minutes=5))
        assert "Resets in now" in str(exc)


# ---------------------------------------------------------------------------
# _session_reset_from_ttl
# ---------------------------------------------------------------------------


class TestSessionResetFromTtl:
    @pytest.mark.asyncio
    async def test_returns_ttl_based_time_when_key_exists(self):
        mock_redis = AsyncMock()
        mock_redis.ttl = AsyncMock(return_value=3600)  # 1 hour

        before = datetime.now(UTC)
        result = await _session_reset_from_ttl(mock_redis, _USER, _SESSION)
        after = datetime.now(UTC)

        # Should be ~1 hour from now
        assert (
            before + timedelta(seconds=3600)
            <= result
            <= after + timedelta(seconds=3600)
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_default_ttl_when_key_expired(self):
        """TTL <= 0 (expired/missing key) should fall back to default TTL."""
        mock_redis = AsyncMock()
        mock_redis.ttl = AsyncMock(return_value=-2)  # Key does not exist

        before = datetime.now(UTC)
        result = await _session_reset_from_ttl(mock_redis, _USER, _SESSION)
        after = datetime.now(UTC)

        expected_min = before + timedelta(seconds=_SESSION_TTL_SECONDS)
        expected_max = after + timedelta(seconds=_SESSION_TTL_SECONDS)
        assert expected_min <= result <= expected_max

    @pytest.mark.asyncio
    async def test_falls_back_to_default_ttl_on_redis_error(self):
        """RedisError should not propagate — falls back to default TTL."""
        mock_redis = AsyncMock()
        mock_redis.ttl = AsyncMock(side_effect=RedisError("Connection lost"))

        before = datetime.now(UTC)
        result = await _session_reset_from_ttl(mock_redis, _USER, _SESSION)
        after = datetime.now(UTC)

        expected_min = before + timedelta(seconds=_SESSION_TTL_SECONDS)
        expected_max = after + timedelta(seconds=_SESSION_TTL_SECONDS)
        assert expected_min <= result <= expected_max


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
    async def test_sets_expire_on_both_keys(self):
        """Pipeline should call expire for both session and weekly keys."""
        mock_pipe = self._make_pipeline_mock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_token_usage(
                _USER, _SESSION, prompt_tokens=100, completion_tokens=50
            )

        expire_calls = mock_pipe.expire.call_args_list
        assert len(expire_calls) == 2

        # Session key TTL should match the configured session TTL
        session_ttl = expire_calls[0].args[1]
        assert session_ttl == _SESSION_TTL_SECONDS

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
            await record_token_usage(
                _USER, _SESSION, prompt_tokens=100, completion_tokens=50
            )

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
            await record_token_usage(
                _USER, _SESSION, prompt_tokens=100, completion_tokens=50
            )
