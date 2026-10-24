"""Boundary tests for pre-auth admission and trusted client identity."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, Request
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.api.external import auth_rate_limit as rate_limit
from backend.api.external import middleware


def request(host="192.0.2.10", forwarded=None):
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": (host, 1234)})


@pytest.mark.parametrize(
    "value", ["garbage", "agpt_", "agpt_!abmiddle", "agpt_ab\nmiddle"]
)
def test_malformed_key_has_no_budget(value):
    assert rate_limit.key_head_from_plaintext(value) is None


@pytest.mark.parametrize("host", ["", "unknown", "not-an-ip", "1.2.3.999"])
def test_invalid_peer_has_no_shared_bucket(host):
    assert rate_limit.client_ip_from_request(request(host)) is None


def test_ipv6_identity_is_canonical():
    assert (
        rate_limit.client_ip_from_request(request("2001:0db8:0:0:0:0:0:1"))
        == "2001:db8::1"
    )


def test_raw_forwarded_header_is_ignored():
    assert (
        rate_limit.client_ip_from_request(request(forwarded="198.51.100.9"))
        == "192.0.2.10"
    )


@pytest.mark.parametrize(
    "peer,forwarded,expected",
    [
        ("192.0.2.10", "198.51.100.9", "192.0.2.10"),
        ("127.0.0.1", "198.51.100.9", "198.51.100.9"),
        ("127.0.0.1", "203.0.113.5, 198.51.100.9", "198.51.100.9"),
    ],
)
async def test_uvicorn_trust_boundary(peer, forwarded, expected):
    observed = []

    async def app(scope, receive, send):
        observed.append(rate_limit.client_ip_from_request(Request(scope)))

    proxy = ProxyHeadersMiddleware(app, trusted_hosts=["127.0.0.1"])
    scope = request(peer, forwarded).scope | {"scheme": "http"}
    await proxy(scope, AsyncMock(), AsyncMock())
    assert observed == [expected]


@pytest.mark.parametrize(
    "error", [RedisConnectionError, RedisClusterException, OSError, ValueError]
)
@pytest.mark.parametrize("phase", ["connect", "eval"])
async def test_redis_failure_does_not_block_authentication(mocker, error, phase):
    if phase == "connect":
        mocker.patch.object(
            rate_limit, "get_redis_async", side_effect=error("unavailable")
        )
    else:
        redis = mocker.Mock()
        redis.eval = AsyncMock(side_effect=error("unavailable"))
        mocker.patch.object(rate_limit, "get_redis_async", return_value=redis)
    assert (
        await rate_limit.enforce_api_key_validate_rate_limit(
            request(), "agpt_abcmiddle"
        )
        is None
    )


@pytest.mark.parametrize("error", [OSError, ValueError])
@pytest.mark.parametrize("entry", [middleware.require_api_key, middleware.require_auth])
async def test_limiter_logic_failure_does_not_reach_key_validation(
    mocker, error, entry
):
    redis = mocker.Mock()
    redis.eval = AsyncMock(return_value=rate_limit.AUTH_VALIDATE_IP_MAX_REQUESTS + 1)
    mocker.patch.object(rate_limit, "get_redis_async", return_value=redis)
    mocker.patch.object(
        rate_limit, "record_rate_limit_hit", side_effect=error("metric failure")
    )
    validate = mocker.patch.object(middleware, "validate_api_key", return_value=None)

    with pytest.raises(error, match="metric failure"):
        await entry(request(), api_key="agpt_abcmiddle")

    validate.assert_not_awaited()


@pytest.mark.parametrize("refund", [False, True])
async def test_timeout_is_bounded_and_cancels_the_pending_operation(mocker, refund):
    cancelled = asyncio.Event()

    async def stalled_connection():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    mocker.patch.object(rate_limit, "get_redis_async", new=stalled_connection)
    mocker.patch.object(rate_limit, "AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS", 0.02)
    budget = rate_limit.APIKeyValidationBudget(ip_key="ip", head_key="head")
    operation = (
        rate_limit.refund_api_key_validate_rate_limit(budget)
        if refund
        else rate_limit.enforce_api_key_validate_rate_limit(request(), "agpt_abcmiddle")
    )
    assert await asyncio.wait_for(operation, timeout=0.4) is None
    assert cancelled.is_set()


async def test_refund_failure_preserves_success(mocker):
    mocker.patch.object(
        rate_limit, "get_redis_async", side_effect=RedisConnectionError("down")
    )
    budget = rate_limit.APIKeyValidationBudget(ip_key="ip", head_key="head")
    await rate_limit.refund_api_key_validate_rate_limit(budget)


async def test_no_reservation_needs_no_redis(mocker):
    redis = mocker.patch.object(rate_limit, "get_redis_async")
    await rate_limit.refund_api_key_validate_rate_limit(None)
    redis.assert_not_called()


@pytest.mark.parametrize("entry", [middleware.require_api_key, middleware.require_auth])
async def test_denied_admission_skips_validation(mocker, entry):
    mocker.patch.object(
        middleware,
        "enforce_api_key_validate_rate_limit",
        side_effect=HTTPException(429),
    )
    validate = mocker.patch.object(middleware, "validate_api_key")
    with pytest.raises(HTTPException) as exc:
        await entry(request(), api_key="agpt_abcmiddle")
    assert exc.value.status_code == 429
    validate.assert_not_called()


@pytest.mark.parametrize("entry", [middleware.require_api_key, middleware.require_auth])
async def test_invalid_key_under_budget_returns_401(mocker, entry):
    mocker.patch.object(
        middleware, "enforce_api_key_validate_rate_limit", return_value=None
    )
    mocker.patch.object(middleware, "validate_api_key", return_value=None)
    refund = mocker.patch.object(middleware, "refund_api_key_validate_rate_limit")
    with pytest.raises(HTTPException) as exc:
        await entry(request(), api_key="agpt_abcmiddle")
    assert exc.value.status_code == 401
    refund.assert_not_called()
