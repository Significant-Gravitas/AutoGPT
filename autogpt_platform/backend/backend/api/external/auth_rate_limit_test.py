"""Boundary tests for pre-auth admission and trusted client identity."""

import asyncio
from datetime import UTC, datetime
from typing import Annotated
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException, Request, Security
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException
from uvicorn._types import HTTPScope
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.api.external import auth_rate_limit as rate_limit
from backend.api.external import middleware
from backend.data.auth.base import APIAuthorizationInfo


@pytest.fixture(params=["api-key", "bearer"])
def credential_headers(request):
    if request.param == "api-key":
        return {"X-API-Key": "agpt_abcmiddle"}
    return {"Authorization": "Bearer agpt_abcmiddle"}


@pytest_asyncio.fixture
async def client():
    app = FastAPI()

    @app.get("/auth")
    async def authenticate(
        auth: Annotated[APIAuthorizationInfo, Security(middleware.require_auth)],
    ):
        return {"user_id": auth.user_id}

    transport = httpx.ASGITransport(app, client=("192.0.2.10", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


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
        == "2001:db8::/64"
    )


def test_ipv4_mapped_ipv6_uses_ipv4_budget():
    assert rate_limit.client_ip_from_request(
        request("::ffff:192.0.2.10")
    ) == rate_limit.client_ip_from_request(request("192.0.2.10"))


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
    scope = HTTPScope(
        type="http",
        asgi={"version": "3.0", "spec_version": "2.3"},
        http_version="1.1",
        method="GET",
        scheme="http",
        path="/auth",
        raw_path=b"/auth",
        query_string=b"",
        root_path="",
        headers=[(b"x-forwarded-for", forwarded.encode())],
        client=(peer, 1234),
        server=None,
    )
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
async def test_limiter_logic_failure_does_not_reach_key_validation(
    mocker, error, client, credential_headers
):
    redis = mocker.Mock()
    redis.eval = AsyncMock(
        return_value=rate_limit.AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS + 1
    )
    mocker.patch.object(rate_limit, "get_redis_async", return_value=redis)
    mocker.patch.object(
        rate_limit, "record_rate_limit_hit", side_effect=error("metric failure")
    )
    validate = mocker.patch.object(middleware, "validate_api_key", return_value=None)

    with pytest.raises(error, match="metric failure"):
        await client.get("/auth", headers=credential_headers)

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
    budget = rate_limit.APIKeyValidationBudget(head_key="head")
    operation = (
        rate_limit.refund_api_key_validate_rate_limit(budget)
        if refund
        else rate_limit.enforce_api_key_validate_rate_limit(request(), "agpt_abcmiddle")
    )
    assert await asyncio.wait_for(operation, timeout=5) is None
    assert cancelled.is_set()


async def test_refund_failure_preserves_success(mocker):
    mocker.patch.object(
        rate_limit, "get_redis_async", side_effect=RedisConnectionError("down")
    )
    budget = rate_limit.APIKeyValidationBudget(head_key="head")
    await rate_limit.refund_api_key_validate_rate_limit(budget)


async def test_no_reservation_needs_no_redis(mocker):
    redis = mocker.patch.object(rate_limit, "get_redis_async")
    await rate_limit.refund_api_key_validate_rate_limit(None)
    redis.assert_not_called()


async def test_denied_admission_skips_validation(mocker, client, credential_headers):
    mocker.patch.object(
        middleware,
        "enforce_api_key_validate_rate_limit",
        side_effect=HTTPException(429),
    )
    validate = mocker.patch.object(middleware, "validate_api_key", return_value=None)
    mocker.patch.object(
        middleware,
        "validate_access_token",
        side_effect=middleware.InvalidTokenError("not found"),
    )
    response = await client.get("/auth", headers=credential_headers)
    assert response.status_code == 429
    validate.assert_not_called()


async def test_invalid_key_under_budget_returns_401(mocker, client, credential_headers):
    mocker.patch.object(
        middleware, "enforce_api_key_validate_rate_limit", return_value=None
    )
    mocker.patch.object(middleware, "validate_api_key", return_value=None)
    mocker.patch.object(
        middleware,
        "validate_access_token",
        side_effect=middleware.InvalidTokenError("not found"),
    )
    refund = mocker.patch.object(middleware, "refund_api_key_validate_rate_limit")
    response = await client.get("/auth", headers=credential_headers)
    assert response.status_code == 401
    refund.assert_not_called()


async def test_success_refunds_after_the_owner_check(
    mocker, client, credential_headers
):
    budget = rate_limit.APIKeyValidationBudget(head_key="head")
    admit = mocker.patch.object(
        middleware, "enforce_api_key_validate_rate_limit", return_value=budget
    )
    principal = APIAuthorizationInfo(
        user_id="user-1", scopes=[], type="api_key", created_at=datetime.now(UTC)
    )
    mocker.patch.object(middleware, "validate_api_key", return_value=principal)
    owner_check = mocker.patch.object(middleware, "_refuse_banned_owner")

    async def refund(reservation):
        owner_check.assert_awaited_once_with(principal)
        assert reservation is budget

    refunded = mocker.patch.object(
        middleware, "refund_api_key_validate_rate_limit", side_effect=refund
    )
    response = await client.get("/auth", headers=credential_headers)
    assert response.status_code == 200
    assert admit.await_args.args[1] == "agpt_abcmiddle"
    refunded.assert_awaited_once_with(budget)


async def test_banned_owner_keeps_reservation(mocker, client, credential_headers):
    mocker.patch.object(middleware, "enforce_api_key_validate_rate_limit")
    mocker.patch.object(
        middleware,
        "validate_api_key",
        return_value=APIAuthorizationInfo(
            user_id="banned", scopes=[], type="api_key", created_at=datetime.now(UTC)
        ),
    )
    mocker.patch.object(
        middleware, "_refuse_banned_owner", side_effect=HTTPException(403)
    )
    refund = mocker.patch.object(middleware, "refund_api_key_validate_rate_limit")
    response = await client.get("/auth", headers=credential_headers)
    assert response.status_code == 403
    refund.assert_not_awaited()


async def test_oauth_does_not_reserve_an_api_key_budget(mocker, client):
    admit = mocker.patch.object(middleware, "enforce_api_key_validate_rate_limit")
    mocker.patch.object(
        middleware,
        "validate_access_token",
        side_effect=middleware.InvalidTokenError("bad"),
    )
    response = await client.get(
        "/auth", headers={"Authorization": "Bearer agpt_xt_invalid"}
    )
    assert response.status_code == 401
    admit.assert_not_awaited()


async def test_v2_request_auth_does_not_apply_the_v1_budget(mocker):
    admit = mocker.patch.object(middleware, "enforce_api_key_validate_rate_limit")
    validate = mocker.patch.object(middleware, "validate_api_key", return_value=None)
    scope = request().scope
    for _ in range(2):
        with pytest.raises(HTTPException) as rejected:
            await middleware.resolve_request_auth(scope, "agpt_abcmiddle", None)
        assert rejected.value.status_code == 401
    validate.assert_awaited_once()
    admit.assert_not_awaited()
