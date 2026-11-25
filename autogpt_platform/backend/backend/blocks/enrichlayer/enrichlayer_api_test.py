"""Enrichlayer against an API that is down: after its bounded retries, the
client raises its own error, with the status and what the API said."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from backend.blocks.enrichlayer._api import EnrichlayerAPIException, EnrichlayerClient
from backend.util.request import DEFAULT_RETRY_MAX_ATTEMPTS, Requests
from backend.util.truncate import truncate

# A proxy's or a CDN's error page, far longer than an error message should be.
_ERROR_PAGE = "<html><body>" + "Service Unavailable. " * 400 + "</body></html>"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, said",
    [
        (
            {
                "text": '{"message": "Service unavailable"}',
                "content_type": "application/json",
            },
            "Service unavailable",
        ),
        ({"text": "upstream connect error"}, "upstream connect error"),
        (
            {"text": _ERROR_PAGE, "content_type": "text/html"},
            truncate(_ERROR_PAGE, 300),
        ),
    ],
    ids=["json-message", "plain-text", "html-page"],
)
async def test_a_profile_fetch_against_a_down_api_raises_the_enrichlayer_error(
    answer: dict, said: str
):
    received: list[str] = []

    async def unavailable(request: web.Request) -> web.Response:
        received.append(request.path)
        return web.Response(status=503, **answer)

    app = web.Application()
    app.router.add_get("/profile", unavailable)
    async with TestServer(app, host="127.0.0.1") as server:
        # The client's own Requests, as it builds it, let through to the local
        # server and not made to wait between attempts.
        client = EnrichlayerClient(
            custom_requests=Requests(
                trusted_origins=["127.0.0.1"], raise_for_status=False, retry_max_wait=0
            )
        )
        client.API_BASE_URL = str(server.make_url("")).rstrip("/")
        with pytest.raises(EnrichlayerAPIException) as raised:
            await asyncio.wait_for(
                client.fetch_profile("https://www.linkedin.com/in/someone"), 10
            )

    assert raised.value.status_code == 503
    prefix = "Enrichlayer API request failed (503): "
    assert str(raised.value) == f"{prefix}{said}"
    # The error stays short whatever the answer's size.
    assert len(str(raised.value)) <= len(prefix) + 300
    assert received == ["/profile"] * DEFAULT_RETRY_MAX_ATTEMPTS
