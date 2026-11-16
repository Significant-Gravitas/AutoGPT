"""Unit + integration tests for the external API rate limiter."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission, APIKeyStatus
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException

from backend.api.external import rate_limit
from backend.api.external.middleware import require_auth
from backend.api.external.v1.routes import v1_router
from backend.data import redis_client
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable


class FakeRedis:
    """In-memory counter fake; the live Redis test below validates the Lua."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.expires_at: dict[str, float] = {}
        self._now = 0.0

    async def eval(self, script: str, numkeys: int, key: str, ttl: str) -> int:
        assert script == rate_limit._INCR_OPEN_WINDOW
        assert numkeys == 1
        count = self.counters.get(key, 0) + 1
        self.counters[key] = count
        if count == 1:
            self.expires_at[key] = self._now + int(ttl)
        return count


@pytest.fixture(autouse=True)
def fixed_window(mocker):
    clock = mocker.patch.object(rate_limit, "datetime")
    clock.now.return_value = datetime(2026, 10, 30, 12, 0, 30, tzinfo=UTC)


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
        return OAuthAccessTokenInfo(
            id=uuid.uuid4().hex,
            application_id="test-app",
            user_id=user_id,
            scopes=list(APIKeyPermission),
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
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
    (key,) = fake_redis.expires_at
    assert key.startswith("external_api:rl:u1:read:")
    # The key ends with the wall-clock window start, so the counter, the TTL
    # and the reported reset all describe the same window.
    window_start = int(key.rsplit(":", 1)[1])
    assert window_start % rate_limit.READ_WINDOW_SECONDS == 0
    assert int(response.headers["X-RateLimit-Reset"]) == (
        window_start + rate_limit.READ_WINDOW_SECONDS
    )
    assert (
        fake_redis.expires_at[key] - fake_redis._now == rate_limit.READ_WINDOW_SECONDS
    )


@pytest.mark.asyncio
async def test_api_keys_of_one_user_share_a_bucket(fake_redis):
    """A second API key must not buy a fresh budget.

    The user is the unit that cannot be multiplied: if the bucket were keyed on
    the credential id, minting another key would reset the limit.
    """
    dep = rate_limit.require_rate_limit(
        max_requests=rate_limit.READ_MAX_REQUESTS,
        window_seconds=rate_limit.READ_WINDOW_SECONDS,
        scope="read",
    )
    response = Response()
    await dep(response=response, auth=_auth(user_id="u1", key_id="k1"))
    await dep(response=response, auth=_auth(user_id="u1", key_id="k2"))

    assert len(fake_redis.counters) == 1
    (key,) = fake_redis.counters
    assert key.startswith("external_api:rl:u1:read:")
    assert fake_redis.counters[key] == 2


@pytest.mark.asyncio
async def test_oauth_refresh_shares_the_bucket(fake_redis):
    """Refreshing an OAuth token must not reset the budget either.

    OAuth principals carry an access-token id, which changes on every refresh;
    the bucket has to follow the user instead.
    """
    dep = rate_limit.require_rate_limit(
        max_requests=rate_limit.READ_MAX_REQUESTS,
        window_seconds=rate_limit.READ_WINDOW_SECONDS,
        scope="read",
    )
    response = Response()
    await dep(response=response, auth=_auth(user_id="u9", key_id=None))
    await dep(response=response, auth=_auth(user_id="u9", key_id=None))

    assert len(fake_redis.counters) == 1
    (key,) = fake_redis.counters
    assert key.startswith("external_api:rl:u9:read:")


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
    app.include_router(v1_router, prefix="/v1")

    async def fake_require_auth() -> APIAuthorizationInfo:
        return _auth()

    app.dependency_overrides[require_auth] = fake_require_auth
    mock_redis.eval = AsyncMock(return_value=rate_limit.READ_MAX_REQUESTS + 1)

    with TestClient(app) as client:
        res = client.get("/v1/me")
    assert res.status_code == 429
    assert res.headers["Retry-After"] is not None
    assert res.headers["X-RateLimit-Limit"] == str(rate_limit.READ_MAX_REQUESTS)
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)
async def test_the_script_opens_a_window_once_on_real_redis():
    """FakeRedis re-implements the Lua, so only real Redis can catch a
    script that never expires the key or re-opens the window on every hit."""
    key = f"external_api:rl:test:{uuid.uuid4().hex}"
    redis = await get_redis_async()
    try:
        assert await rate_limit._incr_window(key, 60) == 1
        assert 0 < await redis.ttl(key) <= 60
        await redis.expire(key, 30)
        assert await rate_limit._incr_window(key, 60) == 2
        assert 0 < await redis.ttl(key) <= 30
    finally:
        await redis.delete(key)


@pytest.mark.asyncio
async def test_a_hung_redis_fails_open_within_the_deadline(mocker, mock_redis):
    """A Redis that never answers must not park the request: the 0.25s
    deadline fires and the request is let through."""

    async def hang(*_):
        await asyncio.Event().wait()

    mock_redis.eval = hang
    wait_for = mocker.spy(asyncio, "wait_for")
    ret, response = await asyncio.wait_for(_call_dependency(), timeout=2)

    assert ret is not None
    assert "X-RateLimit-Limit" not in response.headers
    assert any(call.kwargs.get("timeout") == 0.25 for call in wait_for.call_args_list)


@pytest.mark.asyncio
async def test_request_cancellation_propagates(mock_redis):
    mock_redis.eval.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await _call_dependency()
