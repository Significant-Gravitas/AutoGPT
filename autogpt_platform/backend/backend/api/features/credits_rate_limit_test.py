"""Unit tests for the GET /credits/subscription per-user rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.api.features import credits_rate_limit as rate_limit


@pytest.fixture
def fake_redis(mocker):
    """Patch ``get_redis_async`` to return a MagicMock whose awaitable ``eval``
    returns the current counter value, so each test can drive the count."""
    redis = MagicMock()
    redis.eval = AsyncMock()
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_first_hit_runs_atomic_counter(fake_redis):
    """The counter is a single atomic EVAL (INCR + conditional EXPIRE), so a
    freshly-created key always gets its TTL in the same round-trip — there is
    no window between creation and increment where the key could lack a TTL."""
    fake_redis.eval.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("u1")
    fake_redis.eval.assert_awaited_once()
    args = fake_redis.eval.await_args.args
    # eval(script, numkeys, key, ttl)
    assert args[0] == rate_limit._INCR_OPEN_WINDOW
    assert args[1] == 1
    assert "u1" in args[2]
    assert args[3] == str(rate_limit.SUBSCRIPTION_STATUS_WINDOW_SECONDS)


@pytest.mark.asyncio
async def test_at_limit_passes(fake_redis):
    """Exactly MAX is still allowed — the cap is exclusive (count > MAX raises)."""
    fake_redis.eval.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_over_limit_raises_429(fake_redis):
    """One past the cap raises HTTP 429 with a descriptive detail."""
    fake_redis.eval.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS + 1
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    assert exc_info.value.status_code == 429
    assert str(rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS) in str(
        exc_info.value.detail
    )


@pytest.mark.asyncio
async def test_fails_open_on_transient_redis_error(mocker):
    """A transient Redis failure must not block the billing status read for
    every user — fail open (no exception raised to the caller)."""
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(side_effect=RedisConnectionError("redis down")),
    )
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_non_transient_error_propagates(mocker):
    """A non-transient error (a bug, not a Redis blip) must surface rather than
    silently disabling the limiter."""
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(side_effect=RuntimeError("programming error")),
    )
    with pytest.raises(RuntimeError):
        await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_per_user_keys_are_distinct(fake_redis):
    """The window key derives from ``user_id`` so two users' counters
    never collide."""
    fake_redis.eval.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("alice")
    key_a = fake_redis.eval.await_args.args[2]

    fake_redis.eval.reset_mock()
    fake_redis.eval.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("bob")
    key_b = fake_redis.eval.await_args.args[2]

    assert key_a != key_b
    assert "alice" in key_a
    assert "bob" in key_b
