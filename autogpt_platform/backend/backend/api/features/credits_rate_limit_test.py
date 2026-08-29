"""Unit tests for the GET /credits/subscription per-user rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest

from backend.api.features import credits_rate_limit as rate_limit


@pytest.fixture
def fake_redis(mocker):
    """Patch ``get_redis_async`` to return a MagicMock with awaitable
    ``set`` and ``incr`` so each test can drive the counter directly."""
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.incr = AsyncMock()
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_first_hit_creates_key_with_ttl(fake_redis):
    """``SET NX EX`` runs on every hit and sets the TTL exactly once when
    the window opens (subsequent hits no-op the SET)."""
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("u1")
    fake_redis.set.assert_awaited_once()
    args, kwargs = fake_redis.set.await_args
    key = args[0]
    assert kwargs["ex"] == rate_limit.SUBSCRIPTION_STATUS_WINDOW_SECONDS
    assert kwargs["nx"] is True
    assert "u1" in key


@pytest.mark.asyncio
async def test_at_limit_passes(fake_redis):
    """Exactly MAX is still allowed — the cap is exclusive (count > MAX raises)."""
    fake_redis.incr.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_over_limit_raises_429(fake_redis):
    """One past the cap raises HTTP 429 with a descriptive detail."""
    fake_redis.incr.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS + 1
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    assert exc_info.value.status_code == 429
    assert str(rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS) in str(
        exc_info.value.detail
    )


@pytest.mark.asyncio
async def test_fails_open_on_redis_error(mocker):
    """A Redis brown-out must not block the billing status read — fail-open."""
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    )
    # Should NOT raise.
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_per_user_keys_are_distinct(fake_redis):
    """The window key derives from ``user_id`` so two users' counters
    never collide."""
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("alice")
    key_a = fake_redis.incr.await_args.args[0]

    fake_redis.incr.reset_mock()
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("bob")
    key_b = fake_redis.incr.await_args.args[0]

    assert key_a != key_b
    assert "alice" in key_a
    assert "bob" in key_b
