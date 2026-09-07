"""Rate limiting as a contract: what a client is told, and what it is keyed on.

A cap the caller cannot see forces it to retry blind, and an anonymous bucket
keyed on a header the caller writes is not a cap at all.
"""

from unittest import mock

import pytest
import pytest_mock
from fastapi import HTTPException

from backend.api.external.v2 import credits
from backend.api.external.v2.global_rate_limit import (
    GlobalRateLimitMiddleware,
    client_ip,
)
from backend.api.utils.rate_limit import RateLimiter

PEER = "10.0.0.9"


async def test_a_request_under_the_cap_is_told_where_it_stands(
    redis: mock.AsyncMock,
) -> None:
    redis.incr.return_value = 3

    state = await RateLimiter("t", max_requests=10, window_seconds=60).check("u1")

    assert state is not None
    assert state.headers()["X-RateLimit-Limit"] == "10"
    assert state.headers()["X-RateLimit-Remaining"] == "7"
    assert 1 <= int(state.headers()["X-RateLimit-Reset"]) <= 60


async def test_a_blocked_request_carries_retry_after(redis: mock.AsyncMock) -> None:
    """Without it a client retries blind, adding load to what the cap protects."""
    redis.incr.return_value = 11

    with pytest.raises(HTTPException) as raised:
        await RateLimiter("t", max_requests=10, window_seconds=60).check("u1")

    headers = raised.value.headers or {}
    assert raised.value.status_code == 429
    assert 1 <= int(headers["Retry-After"]) <= 60
    assert headers["X-RateLimit-Remaining"] == "0"


async def test_an_unmeasurable_window_publishes_no_numbers(
    redis: mock.AsyncMock,
) -> None:
    """Redis is down: fail open, but do not report a count nobody measured."""
    redis.incr.side_effect = ConnectionError("redis is gone")

    assert (
        await RateLimiter("t", max_requests=10, window_seconds=60).check("u1") is None
    )


async def test_the_response_carries_the_callers_window_position(
    redis: mock.AsyncMock,
) -> None:
    redis.incr.return_value = 1
    sent: list[dict] = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        sent.append(message)

    await GlobalRateLimitMiddleware(app)(_scope(), _receive, send)

    headers = dict(sent[0]["headers"])
    assert headers[b"x-ratelimit-limit"] == b"5"
    assert headers[b"x-ratelimit-remaining"] == b"4"


@pytest.mark.parametrize(
    "hops, forwarded, expected",
    [
        (1, "203.0.113.7", "203.0.113.7"),
        # The spoofed entry sits left of the one our proxy appended.
        (1, "1.2.3.4, 203.0.113.7", "203.0.113.7"),
        (2, "203.0.113.7, 198.51.100.1", "203.0.113.7"),
        (2, "9.9.9.9, 203.0.113.7, 198.51.100.1", "203.0.113.7"),
        # Fewer hops than configured means the header did not come through
        # our own proxies; the socket peer is the only trustworthy value.
        (2, "203.0.113.7", PEER),
        (1, "", PEER),
        (0, "203.0.113.7", PEER),
    ],
)
def test_the_anonymous_bucket_ignores_caller_written_hops(
    mocker: pytest_mock.MockFixture, hops: int, forwarded: str, expected: str
) -> None:
    """A caller that can set the key picks its own bucket, so there is no cap."""
    mocker.patch(
        "backend.api.external.v2.global_rate_limit.settings.config.trusted_proxy_count",
        hops,
    )

    headers = {b"x-forwarded-for": forwarded.encode()} if forwarded else {}
    assert client_ip(_scope(), headers) == expected


async def test_the_subscription_read_is_capped_before_it_reaches_stripe(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Uncached Stripe reads on every call, so the cap has to gate the fan-out."""
    user = mocker.patch.object(credits, "get_user_by_id", new=mock.AsyncMock())
    mocker.patch.object(
        credits.subscription_limiter,
        "check",
        new=mock.AsyncMock(side_effect=HTTPException(status_code=429, detail="nope")),
    )

    with pytest.raises(HTTPException) as raised:
        await credits.get_subscription_status(auth=mock.Mock(user_id="u1"))

    assert raised.value.status_code == 429
    user.assert_not_awaited()


@pytest.fixture
def redis(mocker: pytest_mock.MockFixture) -> mock.AsyncMock:
    client = mock.AsyncMock()
    client.set.return_value = True
    mocker.patch(
        "backend.api.utils.rate_limit.get_redis_async",
        new=mock.AsyncMock(return_value=client),
    )
    mocker.patch(
        "backend.api.external.v2.global_rate_limit.resolve_auth_info",
        new=mock.AsyncMock(return_value=None),
    )
    return client


def _scope() -> dict:
    return {"type": "http", "client": (PEER, 4242), "headers": []}


async def _receive() -> dict:
    return {"type": "http.request"}
