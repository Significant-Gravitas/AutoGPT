"""Rate limiting as a contract: what a client is told, and what it is keyed on.

A cap the caller cannot see forces it to retry blind, and an anonymous bucket
keyed on a header the caller writes is not a cap at all.
"""

from datetime import UTC, datetime
from unittest import mock

import pytest
import pytest_mock
from fastapi import HTTPException, Response

from backend.api.external.middleware import resolve_request_auth
from backend.api.external.v2 import credits, global_rate_limit
from backend.api.external.v2.global_rate_limit import (
    GlobalRateLimitMiddleware,
    client_ip,
)
from backend.api.utils.rate_limit import RateLimiter
from backend.data.auth.base import APIAuthorizationInfo

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
        await credits.get_subscription_status(
            response=Response(), auth=mock.Mock(user_id="u1")
        )

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
        "backend.api.external.v2.global_rate_limit.resolve_request_auth",
        new=mock.AsyncMock(return_value=None),
    )
    return client


def _scope() -> dict:
    return {"type": "http", "client": (PEER, 4242), "headers": []}


async def _receive() -> dict:
    return {"type": "http.request"}


class _Counters:
    """A fake Redis window store: counts per key, readable as the limiter reads them."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def set(self, name: str, value: int, ex: int = 0, nx: bool = False) -> bool:
        if nx and name in self.counts:
            return False
        self.counts[name] = int(value)
        return True

    async def incr(self, name: str) -> int:
        self.counts[name] = self.counts.get(name, 0) + 1
        return self.counts[name]

    async def get(self, name: str) -> bytes | None:
        return str(self.counts[name]).encode() if name in self.counts else None

    def total(self, fragment: str) -> int:
        return sum(n for key, n in self.counts.items() if fragment in key)


@pytest.fixture
def counters(mocker: pytest_mock.MockFixture) -> _Counters:
    store = _Counters()
    mocker.patch(
        "backend.api.utils.rate_limit.get_redis_async",
        new=mock.AsyncMock(return_value=store),
    )
    return store


async def _send_with_key(key: bytes, verify: mock.AsyncMock) -> int:
    """One request carrying `key`; the status the middleware answered with."""
    statuses: list[int] = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    scope = _scope()
    scope["headers"] = [(b"x-api-key", key)]
    await GlobalRateLimitMiddleware(app)(scope, _receive, send)
    return statuses[0]


async def test_a_valid_key_flooding_the_api_is_refused_before_the_hash(
    mocker: pytest_mock.MockFixture, counters: _Counters
) -> None:
    """The per-user cap answers only after the hash it exists to bound."""
    verify = mocker.patch(
        "backend.api.external.v2.global_rate_limit.resolve_request_auth",
        new=mock.AsyncMock(return_value=mock.Mock(user_id="user-1")),
    )
    mocker.patch.object(
        global_rate_limit._authenticated_limiter, "check", return_value=None
    )

    statuses = [await _send_with_key(b"agpt_validkey", verify) for _ in range(310)]

    assert statuses.count(200) == 300
    assert statuses[-1] == 429
    assert verify.await_count == 300


async def test_a_failing_key_is_refused_unhashed_without_locking_out_other_keys(
    mocker: pytest_mock.MockFixture, counters: _Counters
) -> None:
    """A bad key behind a shared address throttles its own head, not the address."""
    rejection = HTTPException(status_code=401, detail="Invalid API key")

    async def resolve(scope, api_key, bearer):
        if api_key.startswith("agpt_bad"):
            raise rejection
        return mock.Mock(user_id="user-1")

    verify = mocker.patch(
        "backend.api.external.v2.global_rate_limit.resolve_request_auth",
        new=mock.AsyncMock(side_effect=resolve),
    )
    mocker.patch.object(
        global_rate_limit._authenticated_limiter, "check", return_value=None
    )

    for _ in range(30):
        await _send_with_key(b"agpt_badkey", verify)
    hashed = verify.await_count

    assert await _send_with_key(b"agpt_badkey", verify) == 429
    assert verify.await_count == hashed
    assert await _send_with_key(b"agpt_goodkey", verify) == 200


async def test_an_address_failing_across_many_heads_is_refused_unhashed(
    mocker: pytest_mock.MockFixture, counters: _Counters
) -> None:
    verify = mocker.patch(
        "backend.api.external.v2.global_rate_limit.resolve_request_auth",
        new=mock.AsyncMock(side_effect=HTTPException(status_code=401, detail="no")),
    )
    counters.counts[
        global_rate_limit._failed_auth_limiter._key(PEER, datetime.now(UTC))
    ] = 300

    assert await _send_with_key(b"agpt_fresh", verify) == 429
    verify.assert_not_awaited()


async def test_an_oauth_token_is_not_counted_as_a_key_presentation(
    mocker: pytest_mock.MockFixture, counters: _Counters
) -> None:
    """Access tokens are looked up by digest, not hashed, so they skip the counter."""
    mocker.patch(
        "backend.api.external.v2.global_rate_limit.resolve_request_auth",
        new=mock.AsyncMock(return_value=mock.Mock(user_id="user-1")),
    )
    mocker.patch.object(
        global_rate_limit._authenticated_limiter, "check", return_value=None
    )

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        pass

    scope = _scope()
    scope["headers"] = [(b"authorization", b"Bearer agpt_xt_token")]
    await GlobalRateLimitMiddleware(app)(scope, _receive, send)

    assert counters.total("key-presented") == 0


async def test_a_request_verifies_its_credential_once_even_when_rejected(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The rate limiter and the route both ask; an invalid key is hashed once."""
    validate = mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=mock.AsyncMock(return_value=None),
    )
    scope: dict = {"type": "http"}

    for _ in range(2):
        with pytest.raises(HTTPException) as raised:
            await resolve_request_auth(scope, api_key="agpt_wrongkey", bearer=None)
        assert raised.value.status_code == 401

    validate.assert_awaited_once()


async def test_a_cached_verification_is_reused_only_for_the_same_credential(
    mocker: pytest_mock.MockFixture,
) -> None:
    validate = mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new=mock.AsyncMock(side_effect=[_principal("a"), _principal("b")]),
    )
    scope: dict = {"type": "http"}

    first = await resolve_request_auth(scope, api_key="agpt_a", bearer=None)
    again = await resolve_request_auth(scope, api_key="agpt_a", bearer=None)
    other = await resolve_request_auth(scope, api_key="agpt_b", bearer=None)

    assert first is again
    assert other.user_id == "b"
    assert validate.await_count == 2


@pytest.mark.parametrize(
    "path, bucket",
    [
        ("/external-api/v2/openapi.json", "anon-docs:"),
        ("/external-api/v2/docs", "anon-docs:"),
        ("/external-api/v2/library/agents", "anon:"),
    ],
)
async def test_reading_the_docs_without_a_key_has_its_own_bucket(
    redis: mock.AsyncMock, path: str, bucket: str
) -> None:
    """The docs send agents to the spec before they have a key; reading it must
    not use up the five calls a minute everything else anonymous gets."""
    redis.incr.return_value = 1

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        pass

    scope = _scope()
    scope.update(path=path, root_path="/external-api/v2")
    await GlobalRateLimitMiddleware(app)(scope, _receive, send)

    assert redis.incr.await_args.args[0].startswith(f"rl:v2:global:{bucket}")


def _principal(user_id: str) -> APIAuthorizationInfo:
    return APIAuthorizationInfo(
        user_id=user_id, scopes=[], type="api_key", created_at=datetime.now(UTC)
    )
