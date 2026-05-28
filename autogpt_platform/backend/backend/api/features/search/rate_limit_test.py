"""Unit tests for the /search/global per-user QPS rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest

from backend.api.features.search import rate_limit


@pytest.fixture
def fake_redis(mocker):
    """Patch ``get_redis_async`` to return a MagicMock with awaitable
    ``incr`` and ``expire`` so each test can drive the counter directly.
    """
    redis = MagicMock()
    redis.incr = AsyncMock()
    redis.expire = AsyncMock()
    mocker.patch(
        "backend.api.features.search.rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_first_hit_sets_expire(fake_redis):
    """The first hit in a window should set the TTL — subsequent hits
    inside the same window must not touch ``expire``."""
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_global_search_rate_limit("u1")
    fake_redis.expire.assert_awaited_once()
    # Window key bucket aligns to the configured window seconds.
    key, ttl = fake_redis.expire.await_args.args
    assert ttl == rate_limit.GLOBAL_SEARCH_WINDOW_SECONDS
    assert "u1" in key


@pytest.mark.asyncio
async def test_subsequent_hit_skips_expire(fake_redis):
    """``incr`` returning > 1 means the key already exists; don't reset
    the TTL — that would create a sliding-window effect we explicitly
    don't want for this fixed-window design."""
    fake_redis.incr.return_value = 2
    await rate_limit.enforce_global_search_rate_limit("u1")
    fake_redis.expire.assert_not_awaited()


@pytest.mark.asyncio
async def test_under_limit_passes(fake_redis):
    """At-the-limit (exactly MAX) is still allowed — the cap is
    exclusive (count > MAX raises)."""
    fake_redis.incr.return_value = rate_limit.GLOBAL_SEARCH_MAX_REQUESTS
    await rate_limit.enforce_global_search_rate_limit("u1")


@pytest.mark.asyncio
async def test_over_limit_raises_429(fake_redis):
    """One past the cap raises HTTP 429 with a descriptive detail."""
    fake_redis.incr.return_value = rate_limit.GLOBAL_SEARCH_MAX_REQUESTS + 1
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_global_search_rate_limit("u1")
    assert exc_info.value.status_code == 429
    detail = str(exc_info.value.detail)
    assert str(rate_limit.GLOBAL_SEARCH_MAX_REQUESTS) in detail


@pytest.mark.asyncio
async def test_fails_open_on_redis_error(mocker):
    """Redis brown-out must not block search — fail-open keeps everyone
    moving while a small burst of one user's embedding spend is the only
    downside. Caller never sees an exception in this path."""
    mocker.patch(
        "backend.api.features.search.rate_limit.get_redis_async",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    )
    # Should NOT raise.
    await rate_limit.enforce_global_search_rate_limit("u1")


@pytest.mark.asyncio
async def test_per_user_keys_are_distinct(fake_redis):
    """The window key must be derived from ``user_id`` so two users'
    counters never collide. We can't observe the key directly from
    ``incr``'s mock without inspecting args."""
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_global_search_rate_limit("alice")
    key_a = fake_redis.incr.await_args.args[0]

    fake_redis.incr.reset_mock()
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_global_search_rate_limit("bob")
    key_b = fake_redis.incr.await_args.args[0]

    assert key_a != key_b
    assert "alice" in key_a
    assert "bob" in key_b
