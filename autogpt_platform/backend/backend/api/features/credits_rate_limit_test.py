"""Unit tests for the GET /credits/subscription per-user rate limiter."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException, ResponseError

from backend.api.features import credits_rate_limit as rate_limit


class FakeRedis:
    """Minimal Redis stand-in that actually interprets the limiter's Lua.

    ``eval`` is mocked in most suites, which means the script — the only
    non-trivial logic in the module — never runs. This executes the same
    semantics (INCR, and EXPIRE only on the call that opened the window)
    against real state, so a regression in the script (e.g. dropping the
    EXPIRE branch, or refreshing the TTL on every hit) fails a test.
    """

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    async def eval(self, script: str, numkeys: int, key: str, ttl: str) -> int:
        assert script == rate_limit._INCR_OPEN_WINDOW
        assert numkeys == 1
        count = self.counters.get(key, 0) + 1
        self.counters[key] = count
        if count == 1:
            self.ttls[key] = int(ttl)
        return count


@pytest.fixture
def fake_redis(mocker):
    """Patch ``get_redis_async`` to return a stateful fake whose ``eval``
    executes the limiter's script semantics."""
    redis = FakeRedis()
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.fixture
def mock_redis(mocker):
    """Patch ``get_redis_async`` with a MagicMock whose ``eval`` return value
    each test sets directly, for driving specific counts/errors."""
    redis = MagicMock()
    redis.eval = AsyncMock()
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_first_hit_sets_ttl(fake_redis):
    """The window-opening request must set the TTL, or the key would never
    expire and the user would stay blocked forever after hitting the cap."""
    await rate_limit.enforce_subscription_status_rate_limit("u1")
    (key,) = fake_redis.ttls
    assert "u1" in key
    assert fake_redis.ttls[key] == rate_limit.SUBSCRIPTION_STATUS_WINDOW_SECONDS
    assert fake_redis.counters[key] == 1


@pytest.mark.asyncio
async def test_ttl_not_refreshed_on_later_hits(fake_redis):
    """The window is fixed: later hits increment but must not extend the TTL,
    otherwise a steady stream of requests would hold the window open."""
    for _ in range(5):
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    (key,) = fake_redis.ttls
    assert fake_redis.counters[key] == 5
    # Exactly one key, whose TTL was written once when the window opened.
    assert list(fake_redis.ttls.values()) == [
        rate_limit.SUBSCRIPTION_STATUS_WINDOW_SECONDS
    ]


@pytest.mark.asyncio
async def test_window_rolls_over(fake_redis, mocker):
    """A later window uses a different key, so the count starts fresh."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    mocker.patch.object(rate_limit, "datetime", MagicMock(now=lambda tz: base))
    await rate_limit.enforce_subscription_status_rate_limit("u1")

    later = datetime(2026, 1, 1, 12, 1, 30, tzinfo=UTC)
    mocker.patch.object(rate_limit, "datetime", MagicMock(now=lambda tz: later))
    await rate_limit.enforce_subscription_status_rate_limit("u1")

    assert len(fake_redis.counters) == 2
    assert set(fake_redis.counters.values()) == {1}


@pytest.mark.asyncio
async def test_cap_is_enforced_over_a_real_sequence(fake_redis):
    """Exactly MAX requests pass; the next one is rejected."""
    for _ in range(rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS):
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_over_limit_sets_retry_after(mock_redis):
    """429 must carry Retry-After: without it clients retry blind and a single
    block turns into several more requests against the endpoint."""
    mock_redis.eval.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS + 1
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    headers = exc_info.value.headers or {}
    retry_after = int(headers["Retry-After"])
    assert 0 < retry_after <= rate_limit.SUBSCRIPTION_STATUS_WINDOW_SECONDS


@pytest.mark.asyncio
async def test_at_limit_passes(mock_redis):
    """Exactly MAX is still allowed — the cap is exclusive (count > MAX raises)."""
    mock_redis.eval.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.parametrize(
    "error",
    [
        RedisConnectionError("connection refused"),
        # Does not inherit from RedisError; raised during a cluster rolling
        # restart / slot migration. Must not 500 the billing endpoint.
        RedisClusterException("SlotNotCoveredError"),
        ResponseError("MISCONF Redis is configured to save RDB snapshots"),
        OSError("socket blew up"),
    ],
    ids=["connection", "cluster", "response", "os"],
)
@pytest.mark.asyncio
async def test_fails_open_on_redis_errors_from_the_command(mock_redis, error):
    """Every flavour of Redis trouble on an established client must fail open.

    The client is cached per event loop, so in production the connect path
    rarely runs — the real surface is ``eval`` raising mid-command.
    """
    mock_redis.eval.side_effect = error
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_fails_open_when_connect_fails(mocker):
    """Trouble reaching Redis at all also fails open."""
    mocker.patch(
        "backend.api.features.credits_rate_limit.get_redis_async",
        new=AsyncMock(side_effect=RedisClusterException("cluster down")),
    )
    await rate_limit.enforce_subscription_status_rate_limit("u1")


@pytest.mark.asyncio
async def test_fails_open_and_fast_when_redis_hangs(mocker):
    """A hung Redis must not hold the request: the limiter has a deadline and
    falls through instead of parking a worker slot for the retry ladder."""

    async def never_returns(_key: str) -> int:
        await asyncio.sleep(60)
        return 1

    # Replace the function outright rather than mocking it: an AsyncMock whose
    # side_effect returns a coroutine hands that coroutine back as the result
    # instead of awaiting it, so the deadline would never be exercised.
    mocker.patch.object(rate_limit, "_incr_window", new=never_returns)
    mocker.patch.object(rate_limit, "SUBSCRIPTION_STATUS_REDIS_TIMEOUT_SECONDS", 0.05)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await rate_limit.enforce_subscription_status_rate_limit("u1")
    # Tight bound: the patched deadline is 50ms, so anything approaching a
    # second means the timeout was not honoured. Generous enough to absorb CI
    # scheduling jitter, strict enough to fail if the deadline is dropped.
    assert loop.time() - started < 1


@pytest.mark.asyncio
async def test_per_user_keys_are_distinct(fake_redis):
    """The window key derives from ``user_id`` so two users' counters
    never collide."""
    await rate_limit.enforce_subscription_status_rate_limit("alice")
    await rate_limit.enforce_subscription_status_rate_limit("bob")
    keys = sorted(fake_redis.counters)
    assert len(keys) == 2
    assert any("alice" in k for k in keys)
    assert any("bob" in k for k in keys)


def _counter(name: str, **labels) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.mark.asyncio
async def test_over_limit_records_a_rate_limit_hit(mock_redis):
    """The 429 must be visible to Prometheus; the counter had no callers."""
    before = _counter(
        "autogpt_rate_limit_hits_total", endpoint="/api/credits/subscription"
    )
    mock_redis.eval.return_value = rate_limit.SUBSCRIPTION_STATUS_MAX_REQUESTS + 1
    with pytest.raises(fastapi.HTTPException):
        await rate_limit.enforce_subscription_status_rate_limit("u1")
    assert (
        _counter("autogpt_rate_limit_hits_total", endpoint="/api/credits/subscription")
        == before + 1
    )


@pytest.mark.asyncio
async def test_under_limit_does_not_record(mock_redis):
    before = _counter(
        "autogpt_rate_limit_hits_total", endpoint="/api/credits/subscription"
    )
    mock_redis.eval.return_value = 1
    await rate_limit.enforce_subscription_status_rate_limit("u1")
    assert (
        _counter("autogpt_rate_limit_hits_total", endpoint="/api/credits/subscription")
        == before
    )
