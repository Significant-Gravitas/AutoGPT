# -*- coding: utf-8 -*-
"""Unit + integration tests for the external API rate limiter."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from fastapi import Response
from prisma.enums import APIKeyPermission, APIKeyStatus
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException

from backend.api.external import rate_limit
from backend.api.external.middleware import require_auth
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.base import APIAuthorizationInfo


class FakeRedis:
    """Minimal Redis stand-in that actually interprets the limiter's Lua.

    Mirrors the approach in ``credits_rate_limit_test``: the script (the only
    non-trivial logic in the module) runs against real state, so a regression
    in the script fails a test.
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
    redis = FakeRedis()
    mocker.patch(
        "backend.api.external.rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


@pytest.fixture
def mock_redis(mocker):
    """MagicMock whose ``eval`` return value each test sets directly."""
    redis = MagicMock()
    redis.eval = AsyncMock(return_value=1)
    mocker.patch(
        "backend.api.external.rate_limit.get_redis_async",
        new=AsyncMock(return_value=redis),
    )
    return redis


def _auth(user_id: str = "u1", key_id: str | None = "k1") -> APIAuthorizationInfo:
    if key_id is None:
        # OAuth principals have no key id.
        return APIAuthorizationInfo(
            user_id=user_id,
            scopes=list(APIKeyPermission),
            type="oauth",
            created_at=datetime.now(UTC),
        )
    return APIKeyInfo(
        id=key_id,
        name="test-key",
        head="abcd",
        tail="wxyz",
        status=APIKeyStatus.ACTIVE,
        user_id=user_id,
        scopes=list(APIKeyPermission),
        type="api_key",
        created_at=datetime.now(UTC),
    )


async def _call_dependency(**kwargs):
    """Call ``require_rate_limit``'s dependency directly with a Response."""
    dep = rate_limit.require_rate_limit(
        max_requests=rate_limit.READ_MAX_REQUESTS,
        window_seconds=rate_limit.READ_WINDOW_SECONDS,
        scope="read",
    )
    response = Response()
    auth = kwargs.get("auth", _auth())
    ret = await dep(response=response, auth=auth)
    return ret, response


@pytest.mark.asyncio
async def test_first_hit_sets_ttl_and_headers(fake_redis):
    """The window-opening request must set the TTL and expose X-RateLimit-*
    headers on the response."""
    _, response = await _call_dependency()

    assert response.headers["X-RateLimit-Limit"] == str(rate_limit.READ_MAX_REQUESTS)
    assert response.headers["X-RateLimit-Remaining"] == str(
        rate_limit.READ_MAX_REQUESTS - 1
    )
    assert int(response.headers["X-RateLimit-Reset"]) > 0
    (key,) = fake_redis.ttls
    assert key.startswith("external_api:rl:u1:")
    assert "k1" in key
    assert fake_redis.ttls[key] == rate_limit.READ_WINDOW_SECONDS


@pytest.mark.asyncio
async def test_api_key_gets_independent_bucket(fake_redis):
    """Two API keys of the same user must not share a budget."""
    dep = rate_limit.require_rate_limit(
        max_requests=rate_limit.READ_MAX_REQUESTS,
        window_seconds=rate_limit.READ_WINDOW_SECONDS,
        scope="read",
    )
    response = Response()
    await dep(response=response, auth=_auth(user_id="u1", key_id="k1"))
    await dep(response=response, auth=_auth(user_id="u1", key_id="k2"))

    assert len(fake_redis.counters) == 2
    assert all("k1" in k or "k2" in k for k in fake_redis.counters)


@pytest.mark.asyncio
async def test_over_limit_returns_429(mock_redis):
    """Once the counter exceeds the cap the dependency raises HTTP 429 with
    Retry-After and the rate-limit headers."""
    mock_redis.eval = AsyncMock(return_value=rate_limit.READ_MAX_REQUESTS + 1)
    dep = rate_limit.require_rate_limit(
        max_requests=rate_limit.READ_MAX_REQUESTS,
        window_seconds=rate_limit.READ_WINDOW_SECONDS,
        scope="read",
    )
    response = Response()
    with pytest.raises(fastapi.HTTPException) as exc_info:
        await dep(response=response, auth=_auth())

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers["Retry-After"] is not None
    assert exc.headers["X-RateLimit-Remaining"] == "0"
    assert exc.headers["X-RateLimit-Limit"] == str(rate_limit.READ_MAX_REQUESTS)


@pytest.mark.asyncio
async def test_redis_failure_fails_open(mock_redis):
    """A Redis blip must not take the request down: log and let it through."""
    mock_redis.eval = AsyncMock(side_effect=RedisConnectionError("connection refused"))
    ret, response = await _call_dependency()

    assert ret is not None
    assert "X-RateLimit-Limit" not in response.headers


@pytest.mark.asyncio
async def test_cluster_exception_fails_open(mock_redis):
    """RedisClusterException does not inherit from RedisError and would
    otherwise surface as 500 — it must fail open too."""
    mock_redis.eval = AsyncMock(side_effect=RedisClusterException("slot missing"))
    ret, _ = await _call_dependency()

    assert ret is not None


def test_integration_429_via_route(mocker, mock_redis):
    """A real request through the router gets a 429 once the limit is hit."""
    app = fastapi.FastAPI()
    from backend.api.external.v1.routes import v1_router

    app.include_router(v1_router, prefix="/v1")

    async def fake_require_auth() -> APIAuthorizationInfo:
        return _auth()

    app.dependency_overrides[require_auth] = fake_require_auth
    mock_redis.eval = AsyncMock(return_value=rate_limit.READ_MAX_REQUESTS + 1)

    client = fastapi.testclient.TestClient(app)
    res = client.get("/v1/me")
    assert res.status_code == 429
    assert res.headers["Retry-After"] is not None
    assert res.headers["X-RateLimit-Limit"] == str(rate_limit.READ_MAX_REQUESTS)
    app.dependency_overrides.clear()
