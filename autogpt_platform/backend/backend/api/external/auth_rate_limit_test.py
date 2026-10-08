"""Unit tests for external API auth-path (pre-Scrypt) rate limiting."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from fastapi import Request
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException

from backend.api.external import auth_rate_limit as rate_limit
from backend.api.external import middleware


class FakeRedis:
    """Stateful stand-in that executes the limiter's Lua semantics."""

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


def _request(ip: str = "203.0.113.10", forwarded: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if forwarded is not None:
        headers.append((b"x-forwarded-for", forwarded.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v1/me",
        "raw_path": b"/v1/me",
        "query_string": b"",
        "headers": headers,
        "client": (ip, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.fixture
def fake_redis(mocker):
    redis = FakeRedis()
    mocker.patch(
        "backend.api.external.auth_rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.fixture
def mock_redis(mocker):
    redis = MagicMock()
    redis.eval = AsyncMock()
    mocker.patch(
        "backend.api.external.auth_rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_key_head_from_plaintext_requires_prefix_and_length():
    assert rate_limit.key_head_from_plaintext("not-a-key") is None
    assert rate_limit.key_head_from_plaintext("agpt_") is None
    assert rate_limit.key_head_from_plaintext("agpt_abcXXXX") == "agpt_abc"


def test_client_ip_ignores_spoofed_forwarded_for():
    """A caller-supplied X-Forwarded-For must not change the limiter identity,
    otherwise rotating the header yields a fresh bucket per request."""
    req = _request(ip="10.0.0.1", forwarded="198.51.100.7, 10.0.0.1")
    assert rate_limit.client_ip_from_request(req) == "10.0.0.1"


@pytest.mark.asyncio
async def test_rotating_forwarded_for_does_not_bypass_head_cap(fake_redis, mocker):
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_MAX_REQUESTS", 2)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_IP_MAX_REQUESTS", 1000)
    key = "agpt_abcXXXXXrest"
    for i in range(2):
        await rate_limit.enforce_api_key_validate_rate_limit(
            _request(forwarded=f"198.51.100.{i}"), key
        )
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_api_key_validate_rate_limit(
            _request(forwarded="198.51.100.99"), key
        )
    assert exc_info.value.status_code == 429


def test_client_ip_falls_back_to_asgi_client():
    req = _request(ip="203.0.113.50")
    assert rate_limit.client_ip_from_request(req) == "203.0.113.50"


# ---------------------------------------------------------------------------
# Limiter behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_hit_sets_ttl(fake_redis):
    await rate_limit.enforce_api_key_validate_rate_limit(
        _request(), "agpt_abcXXXXXrest"
    )
    assert fake_redis.ttls
    assert all(
        ttl == rate_limit.AUTH_VALIDATE_WINDOW_SECONDS
        for ttl in fake_redis.ttls.values()
    )


@pytest.mark.asyncio
async def test_malformed_key_skips_limiter(fake_redis):
    """Wrong-prefix keys never reach Scrypt — do not spend the budget."""
    await rate_limit.enforce_api_key_validate_rate_limit(_request(), "garbage")
    assert fake_redis.counters == {}


@pytest.mark.asyncio
async def test_head_cap_is_enforced(fake_redis, mocker):
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_MAX_REQUESTS", 3)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_IP_MAX_REQUESTS", 1000)
    key = "agpt_abcXXXXXrest"
    req = _request()
    for _ in range(3):
        await rate_limit.enforce_api_key_validate_rate_limit(req, key)
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_api_key_validate_rate_limit(req, key)
    assert exc_info.value.status_code == 429
    assert "Retry-After" in (exc_info.value.headers or {})


@pytest.mark.asyncio
async def test_ip_cap_is_enforced_across_heads(fake_redis, mocker):
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_MAX_REQUESTS", 1000)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_IP_MAX_REQUESTS", 3)
    req = _request()
    for i in range(3):
        await rate_limit.enforce_api_key_validate_rate_limit(
            req, f"agpt_ab{i}XXXXXrest"
        )
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await rate_limit.enforce_api_key_validate_rate_limit(req, "agpt_abzXXXXXrest")
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_distinct_ips_have_independent_budgets(fake_redis, mocker):
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_MAX_REQUESTS", 1)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_IP_MAX_REQUESTS", 1000)
    key = "agpt_abcXXXXXrest"
    await rate_limit.enforce_api_key_validate_rate_limit(_request("1.1.1.1"), key)
    # Same head from another IP still allowed.
    await rate_limit.enforce_api_key_validate_rate_limit(_request("2.2.2.2"), key)
    with pytest.raises(fastapi.HTTPException):
        await rate_limit.enforce_api_key_validate_rate_limit(_request("1.1.1.1"), key)


@pytest.mark.asyncio
async def test_fails_open_on_redis_errors(mock_redis):
    mock_redis.eval.side_effect = RedisConnectionError("down")
    await rate_limit.enforce_api_key_validate_rate_limit(
        _request(), "agpt_abcXXXXXrest"
    )


@pytest.mark.asyncio
async def test_fails_open_on_cluster_exception(mock_redis):
    mock_redis.eval.side_effect = RedisClusterException("SlotNotCoveredError")
    await rate_limit.enforce_api_key_validate_rate_limit(
        _request(), "agpt_abcXXXXXrest"
    )


@pytest.mark.asyncio
async def test_fails_open_and_fast_when_redis_hangs(mocker):
    async def never_returns(_key: str, _window: int) -> int:
        await asyncio.sleep(60)
        return 1

    mocker.patch.object(rate_limit, "_incr_window", new=never_returns)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS", 0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await rate_limit.enforce_api_key_validate_rate_limit(
        _request(), "agpt_abcXXXXXrest"
    )
    assert loop.time() - started < 1


# ---------------------------------------------------------------------------
# Middleware: limiter gates validate_api_key / Scrypt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_api_key_skips_validate_when_rate_limited(mocker):
    """Over-budget callers must not reach validate_api_key (Scrypt)."""
    mocker.patch(
        "backend.api.external.middleware.enforce_api_key_validate_rate_limit",
        new=AsyncMock(
            side_effect=fastapi.HTTPException(status_code=429, detail="limited")
        ),
    )
    validate = mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=AsyncMock(),
    )
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await middleware.require_api_key(_request(), api_key="agpt_abcXXXXXrest")
    assert exc_info.value.status_code == 429
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_auth_skips_validate_when_rate_limited(mocker):
    mocker.patch(
        "backend.api.external.middleware.enforce_api_key_validate_rate_limit",
        new=AsyncMock(
            side_effect=fastapi.HTTPException(status_code=429, detail="limited")
        ),
    )
    validate = mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=AsyncMock(),
    )
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await middleware.require_auth(_request(), api_key="agpt_abcXXXXXrest")
    assert exc_info.value.status_code == 429
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_api_key_legit_key_still_works(mocker):
    """Under the budget, a valid key still authenticates."""
    mocker.patch(
        "backend.api.external.middleware.enforce_api_key_validate_rate_limit",
        new=AsyncMock(return_value=None),
    )
    fake_info = MagicMock()
    mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=AsyncMock(return_value=fake_info),
    )
    result = await middleware.require_api_key(_request(), api_key="agpt_abcXXXXXrest")
    assert result is fake_info


@pytest.mark.asyncio
async def test_require_api_key_invalid_still_401_under_budget(mocker):
    mocker.patch(
        "backend.api.external.middleware.enforce_api_key_validate_rate_limit",
        new=AsyncMock(return_value=None),
    )
    mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=AsyncMock(return_value=None),
    )
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await middleware.require_api_key(_request(), api_key="agpt_abcXXXXXrest")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_abuse_burst_stops_scrypt_after_threshold(fake_redis, mocker):
    """End-to-end: after the head budget, validate_api_key is not called again."""
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_MAX_REQUESTS", 5)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_IP_MAX_REQUESTS", 1000)
    validate = mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=AsyncMock(return_value=None),
    )
    key = "agpt_abcGARBAGErestofkey"
    req = _request()
    for _ in range(5):
        with pytest.raises(fastapi.HTTPException) as exc:
            await middleware.require_api_key(req, api_key=key)
        assert exc.value.status_code == 401

    assert validate.await_count == 5

    with pytest.raises(fastapi.HTTPException) as exc:
        await middleware.require_api_key(req, api_key=key)
    assert exc.value.status_code == 429
    assert validate.await_count == 5  # not hammered past threshold


# ---------------------------------------------------------------------------
# Candidate lookup (no truncation of colliding heads)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_api_key_checks_every_key_sharing_a_head(mocker):
    """Six ACTIVE keys share a head; the valid one is last. It must still
    authenticate (the lookup must not truncate the candidate list)."""
    from backend.data.auth import api_key as api_key_mod

    records = [MagicMock(name=f"rec{i}") for i in range(6)]

    async def find_many(*, where, take=None, **_kwargs):
        return records[:take] if take is not None else list(records)

    prisma_client = MagicMock()
    prisma_client.find_many = AsyncMock(side_effect=find_many)
    mocker.patch.object(api_key_mod.PrismaAPIKey, "prisma", return_value=prisma_client)

    candidates = []
    for i, rec in enumerate(records):
        cand = MagicMock()
        cand.salt = "salt"
        cand.match.return_value = i == 5
        cand.without_hash.return_value = f"info-{i}"
        candidates.append(cand)
    by_rec = dict(zip(map(id, records), candidates))
    mocker.patch.object(
        api_key_mod.APIKeyInfoWithHash,
        "from_db",
        side_effect=lambda rec: by_rec[id(rec)],
    )

    result = await api_key_mod.validate_api_key("agpt_abcVALIDKEYrest")
    assert result == "info-5"
    assert "take" not in prisma_client.find_many.await_args.kwargs
