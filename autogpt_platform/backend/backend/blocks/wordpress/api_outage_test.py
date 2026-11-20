"""The WordPress calls the block tests don't reach, against a WordPress that is
down and answers with a proxy's HTML page: the error carries the page's text,
all of a short page, and the start and end of a long one."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from backend.blocks.wordpress import _api
from backend.util.request import DEFAULT_RETRY_MAX_ATTEMPTS, Requests

ENDS_WITHIN_SECONDS = 10
PAGES = {
    "short-page": "<html>503 Service Unavailable</html>",
    "long-page": "<html><body>" + "Service Unavailable. " * 400 + "</body></html>",
}


class LocalRequests(Requests):
    def __init__(self, **kwargs):
        super().__init__(trusted_origins=["127.0.0.1"], retry_max_wait=0, **kwargs)


@pytest_asyncio.fixture
async def wordpress_down(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    """A WordPress answering every request with one of ``PAGES``; yields the
    page and the requests it got."""
    page = PAGES[request.param]
    received: list[str] = []

    async def unavailable(http_request: web.Request) -> web.Response:
        received.append(http_request.method)
        return web.Response(status=503, text=page, content_type="text/html")

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", unavailable)
    async with TestServer(app, host="127.0.0.1") as server:
        monkeypatch.setattr(_api, "WORDPRESS_BASE_URL", str(server.make_url("/")))
        monkeypatch.setattr(_api, "Requests", LocalRequests)
        yield page, received


CALLS: dict[str, Callable[[], Awaitable[Any]]] = {
    "exchange-code": lambda: _api.oauth_exchange_code_for_tokens(
        "client", "secret", "code", "https://app.example/callback"
    ),
    "refresh-tokens": lambda: _api.oauth_refresh_tokens("client", "secret", "refresh"),
    "validate-token": lambda: _api.validate_token("client", "token"),
    "api-request": lambda: _api.make_api_request("/rest/v1/me/", "token"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("wordpress_down", list(PAGES), indirect=True)
@pytest.mark.parametrize("call", list(CALLS))
async def test_a_call_against_a_wordpress_that_is_down_carries_the_pages_text(
    wordpress_down: tuple[str, list[str]], call: str
):
    page, received = wordpress_down

    with pytest.raises(ValueError) as raised:
        await asyncio.wait_for(CALLS[call](), ENDS_WITHIN_SECONDS)

    message = str(raised.value)
    assert "bound method" not in message
    # Each message ends with the status, then what the page said.
    said = message.split(" 503 ", 1)[1]
    if len(page) <= 300:
        assert said == page
    else:
        # Cut to 300 characters: its start and its end.
        assert len(said) == 300
        assert said.startswith(page[:100]) and said.endswith(page[-100:])
    assert len(received) == DEFAULT_RETRY_MAX_ATTEMPTS
