"""Unit tests for the /search/global per-user QPS rate limiter."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
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


def _counter(name: str, **labels) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.mark.asyncio
async def test_over_limit_records_a_rate_limit_hit(fake_redis):
    before = _counter("autogpt_rate_limit_hits_total", endpoint="/api/search")
    fake_redis.incr.return_value = rate_limit.GLOBAL_SEARCH_MAX_REQUESTS + 1
    with pytest.raises(fastapi.HTTPException):
        await rate_limit.enforce_global_search_rate_limit("u1")
    assert (
        _counter("autogpt_rate_limit_hits_total", endpoint="/api/search") == before + 1
    )


# The store search limits below run against the real Redis: the window and its
# expiry are Redis's to keep, so a double would only restate the code.


@pytest.fixture
def frozen_minute(monkeypatch):
    """Pin every call to 30 s into one minute, so no test straddles a window,
    with room for a cold Redis connection."""

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)

    monkeypatch.setattr(rate_limit, "datetime", _Frozen)
    monkeypatch.setattr(rate_limit, "PUBLIC_SEARCH_REDIS_TIMEOUT_SECONDS", 10)


@pytest.mark.asyncio
async def test_a_store_search_client_past_its_limit_gets_429_with_retry_after(
    frozen_minute, monkeypatch
):
    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_requests_per_minute", 2
    )
    client = f"client-{uuid.uuid4().hex}"
    await rate_limit.enforce_public_search_rate_limit(client, user_id=None)
    await rate_limit.enforce_public_search_rate_limit(client, user_id=None)

    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_public_search_rate_limit(client, user_id=None)

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "30"}
    await rate_limit.enforce_public_search_rate_limit(
        f"other-{uuid.uuid4().hex}", user_id=None
    )


@pytest.mark.asyncio
async def test_store_search_embeds_only_its_budget_per_minute(
    frozen_minute, monkeypatch
):
    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_embeddings_per_minute", 2
    )
    monkeypatch.setattr(
        rate_limit,
        "PUBLIC_SEARCH_EMBEDDINGS_KEY_PREFIX",
        f"test:store:search:embeddings:{uuid.uuid4().hex}",
    )

    slots = [await rate_limit.take_public_search_embedding_slot() for _ in range(3)]

    assert slots == [True, True, False]


@pytest.mark.asyncio
async def test_store_search_reports_its_embedding_budget_once_nearly_spent_and_once_spent(
    frozen_minute, monkeypatch, caplog
):
    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_embeddings_per_minute", 5
    )
    monkeypatch.setattr(
        rate_limit,
        "PUBLIC_SEARCH_EMBEDDINGS_KEY_PREFIX",
        f"test:store:search:embeddings:{uuid.uuid4().hex}",
    )

    errors_after_each_call = []
    with caplog.at_level(logging.ERROR, logger=rate_limit.logger.name):
        for _ in range(8):
            await rate_limit.take_public_search_embedding_slot()
            errors_after_each_call.append(len(_errors(caplog)))

    assert errors_after_each_call == [0, 0, 0, 1, 1, 2, 2, 2]
    assert [vars(r)["json_fields"]["budget_stage"] for r in _errors(caplog)] == [
        "nearly_spent",
        "spent",
    ]


@pytest.mark.asyncio
async def test_a_disabled_embedding_budget_takes_no_slot_and_raises_no_alert(
    frozen_minute, monkeypatch, caplog
):
    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_embeddings_per_minute", 0
    )
    monkeypatch.setattr(
        rate_limit,
        "PUBLIC_SEARCH_EMBEDDINGS_KEY_PREFIX",
        f"test:store:search:embeddings:{uuid.uuid4().hex}",
    )

    with caplog.at_level(logging.ERROR, logger=rate_limit.logger.name):
        slots = [await rate_limit.take_public_search_embedding_slot() for _ in range(3)]

    assert slots == [False, False, False]
    assert _errors(caplog) == []


@pytest.mark.asyncio
async def test_without_redis_store_search_is_not_limited_but_embeds_nothing(mocker):
    mocker.patch.object(
        rate_limit,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("redis down")),
    )

    await rate_limit.enforce_public_search_rate_limit("203.0.113.9", user_id=None)
    assert await rate_limit.take_public_search_embedding_slot() is False


def _errors(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.ERROR]
