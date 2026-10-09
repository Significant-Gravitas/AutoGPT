"""The anonymous bucket is keyed on an address the caller cannot choose."""

import pytest
import pytest_mock

from backend.api.external.v2 import global_rate_limit
from backend.api.external.v2.global_rate_limit import (
    GlobalRateLimitMiddleware,
    client_ip,
)
from backend.util.settings import Config

PEER = "10.0.0.9"
CLIENT = "203.0.113.7"
# What the hosted platform's proxies append after the client: Cloudflare's
# edge address, then Google's load balancer appends that and its own.
OUR_PROXIES = "172.70.1.1, 34.1.1.1"


@pytest.mark.parametrize("spoofed", ["", "6.6.6.6, ", "6.6.6.6, 7.7.7.7, "])
async def test_a_spoofed_hop_does_not_move_the_anonymous_bucket(
    mocker: pytest_mock.MockFixture, spoofed: str
) -> None:
    """A caller that can set the key picks its own bucket, so there is no cap."""
    mocker.patch.object(
        global_rate_limit.settings.config,
        "trusted_proxy_count",
        Config.model_fields["trusted_proxy_count"].default,
    )
    anonymous = mocker.patch.object(global_rate_limit._anonymous_limiter, "check")

    await _call_middleware(f"{spoofed}{CLIENT}, {OUR_PROXIES}")

    anonymous.assert_awaited_once_with(CLIENT)


@pytest.mark.parametrize(
    "hops, forwarded, expected",
    [
        (3, f"{CLIENT}, {OUR_PROXIES}", CLIENT),
        (1, f"1.2.3.4, {CLIENT}", CLIENT),
        # Fewer entries than our proxies append: the request did not come
        # through them, so the socket peer is the only trustworthy value.
        (3, f"{CLIENT}, 34.1.1.1", PEER),
        (1, "", PEER),
        (0, f"{CLIENT}, {OUR_PROXIES}", PEER),
    ],
)
def test_the_client_is_the_entry_our_first_proxy_appended(
    mocker: pytest_mock.MockFixture, hops: int, forwarded: str, expected: str
) -> None:
    mocker.patch.object(global_rate_limit.settings.config, "trusted_proxy_count", hops)

    headers = {b"x-forwarded-for": forwarded.encode()} if forwarded else {}
    assert client_ip({"type": "http", "client": (PEER, 0)}, headers) == expected


async def _call_middleware(forwarded: str) -> None:
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/runs",
        "headers": [(b"x-forwarded-for", forwarded.encode())],
        "client": (PEER, 0),
    }
    await GlobalRateLimitMiddleware(app)(scope, receive, send)
