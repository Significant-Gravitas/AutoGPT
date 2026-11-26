"""The WordPress blocks against a WordPress that is down.

Their client checks statuses itself (``raise_for_status=False``) and sets no
attempt cap, so a 5xx gets the default number of attempts: a block run against
a site that is down fails with the site's error instead of waiting for it,
whatever shape that error's body has.
"""

import asyncio

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import SecretStr

from backend.blocks._base import Block
from backend.blocks.wordpress import _api
from backend.blocks.wordpress.blog import (
    WordPressCreatePostBlock,
    WordPressGetAllPostsBlock,
)
from backend.data.execution import ExecutionContext
from backend.data.model import OAuth2Credentials
from backend.util.exceptions import BlockExecutionError
from backend.util.request import DEFAULT_RETRY_MAX_ATTEMPTS, Requests
from backend.util.truncate import truncate

CREDENTIALS = OAuth2Credentials(
    id="3f2b8a4e-6c1d-4e7f-9a0b-1c2d3e4f5a6b",
    provider="wordpress",
    title="example.wordpress.com",
    access_token=SecretStr("test-wordpress-token"),
    refresh_token=None,
    access_token_expires_at=None,
    refresh_token_expires_at=None,
    scopes=["posts"],
)
CREDENTIALS_INPUT = {
    "provider": CREDENTIALS.provider,
    "id": CREDENTIALS.id,
    "type": CREDENTIALS.type,
    "title": CREDENTIALS.title,
}
SITE = "example.wordpress.com"

# The attempts go to a local server and take milliseconds; this only trips
# when they never stop.
ENDS_WITHIN_SECONDS = 10

# A proxy's or a CDN's error page, far longer than a block's error should be.
_ERROR_PAGE = "<html><body>" + "Service Unavailable. " * 400 + "</body></html>"

# What a WordPress that is down answers: JSON with a message, or what a proxy
# or a CDN in front of it sends instead. Each maps to the text the block's
# error must carry: all of a short answer, and the start and end of a long one.
DOWN = {
    "json-message": (
        {
            "status": 503,
            "text": '{"error": "unavailable", "message": "Down for maintenance"}',
            "content_type": "application/json",
        },
        "503 - Down for maintenance",
    ),
    "html": (
        {
            "status": 503,
            "text": "<html>503 Service Unavailable</html>",
            "content_type": "text/html",
        },
        "503 - <html>503 Service Unavailable</html>",
    ),
    "plain-text": ({"status": 429, "text": "rate limited"}, "429 - rate limited"),
    "empty": ({"status": 502}, "502 - "),
    "html-page": (
        {"status": 503, "text": _ERROR_PAGE, "content_type": "text/html"},
        f"503 - {truncate(_ERROR_PAGE, 300)}",
    ),
}


class LocalRequests(Requests):
    """The real client, as each block builds it, let through to the local
    server and not made to wait between attempts."""

    def __init__(self, **kwargs):
        super().__init__(trusted_origins=["127.0.0.1"], retry_max_wait=0, **kwargs)


@pytest_asyncio.fixture
async def wordpress_down(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    """A WordPress giving every request one of ``DOWN``'s answers; yields the
    requests it got, and the text the block's error must carry."""
    answer, said = DOWN[request.param]
    received: list[str] = []

    async def unavailable(http_request: web.Request) -> web.Response:
        received.append(f"{http_request.method} {http_request.path}")
        return web.Response(**answer)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", unavailable)
    async with TestServer(app, host="127.0.0.1") as server:
        monkeypatch.setattr(_api, "WORDPRESS_BASE_URL", str(server.make_url("/")))
        monkeypatch.setattr(_api, "Requests", LocalRequests)
        yield received, said


@pytest.mark.parametrize("wordpress_down", list(DOWN), indirect=True)
@pytest.mark.parametrize(
    "block_class, input_data, request_line",
    [
        (
            WordPressGetAllPostsBlock,
            {"site": SITE},
            f"GET /rest/v1.1/sites/{SITE}/posts",
        ),
        (
            WordPressCreatePostBlock,
            {"site": SITE, "title": "Hello", "content": "<p>Hello</p>"},
            f"POST /rest/v1.1/sites/{SITE}/posts/new",
        ),
    ],
    ids=["get-all-posts", "create-post"],
)
async def test_a_block_against_a_wordpress_that_is_down_fails_instead_of_hanging(
    wordpress_down: tuple[list[str], str],
    block_class: type[Block],
    input_data: dict,
    request_line: str,
):
    received, said = wordpress_down
    block = block_class()

    async def run() -> list:
        return [
            output
            async for output in block.execute(
                {**input_data, "credentials": CREDENTIALS_INPUT},
                execution_context=ExecutionContext(),
                credentials=CREDENTIALS,
            )
        ]

    with pytest.raises(BlockExecutionError) as raised:
        await asyncio.wait_for(run(), ENDS_WITHIN_SECONDS)

    message = str(raised.value)
    assert said in message
    assert "bound method" not in message
    # The error stays short whatever the answer's size.
    assert len(message) <= 400
    assert received == [request_line] * DEFAULT_RETRY_MAX_ATTEMPTS
