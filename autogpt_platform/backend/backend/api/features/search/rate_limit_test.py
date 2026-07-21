"""Unit tests for the /search/global per-user QPS rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest

from backend.api.features.search import rate_limit


@pytest.fixture
def fake_redis(mocker):
    """Patch ``get_redis_async`` to return a MagicMock with awaitable
    ``set`` and ``incr`` so each test can drive the counter directly.
    """
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.incr = AsyncMock()
    mocker.patch(
        "backend.api.features.search.rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_first_hit_creates_key_with_ttl(fake_redis):
    """``SET NX EX`` is the atomic create-with-TTL that runs on every
    hit — subsequent hits are no-ops on the SET (key already exists),
    so the TTL is set exactly once when the window opens."""
    fake_redis.incr.return_value = 1
    await rate_limit.enforce_global_search_rate_limit("u1")
    fake_redis.set.assert_awaited_once()
    args, kwargs = fake_redis.set.await_args
    key = args[0]
    assert kwargs["ex"] == rate_limit.GLOBAL_SEARCH_WINDOW_SECONDS
    assert kwargs["nx"] is True
    assert "u1" in key


@pytest.mark.asyncio
async def test_subsequent_hit_still_calls_set_nx(fake_redis):
    """Every hit issues ``SET NX EX`` — Redis no-ops the actual write
    when the key already exists, so the TTL stays put. The point of NX
    is to make this safe to call unconditionally."""
    fake_redis.incr.return_value = 2
    await rate_limit.enforce_global_search_rate_limit("u1")
    fake_redis.set.assert_awaited_once()


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
