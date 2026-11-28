import json
import traceback
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from backend.blocks.helpers import http
from backend.blocks.search import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GetWeatherInformationBlock,
)
from backend.util.request import HTTPClientError, HTTPServerError, Requests

KEY = TEST_CREDENTIALS.api_key.get_secret_value()


@pytest.mark.asyncio
async def test_weather_request_carries_the_real_key_over_https(monkeypatch):
    # Answers as OpenWeatherMap does: 401 unless appid is a real key.
    async def respond(incoming: web.Request):
        if incoming.query.get("appid") != KEY:
            return web.json_response(
                {"cod": 401, "message": "Invalid API key"}, status=401
            )
        return web.json_response(
            {
                "main": {"temp": 12.5, "humidity": 81},
                "weather": [{"description": "light rain"}],
            }
        )

    async with _openweathermap(monkeypatch, respond) as requested:
        outputs = await _run("London")

    sent = urlparse(requested[0])
    assert (sent.scheme, sent.hostname) == ("https", "api.openweathermap.org")
    assert outputs == {
        "temperature": "12.5",
        "humidity": "81",
        "condition": "light rain",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, error_type", [(404, HTTPClientError), (502, HTTPServerError)]
)
async def test_a_failed_weather_request_does_not_quote_the_key(
    monkeypatch, status, error_type
):
    body = {"cod": str(status), "message": "city not found"}

    async def refuse(_incoming: web.Request):
        return web.json_response(body, status=status)

    async with _openweathermap(monkeypatch, refuse):
        with pytest.raises(error_type) as raised:
            await _run("Nowhereville")

    assert raised.value.status_code == status
    assert str(raised.value).endswith(f"Body: {json.dumps(body)}")
    # The executor logs a failed node's traceback, chained causes included.
    assert KEY not in "".join(traceback.format_exception(raised.value))


async def _run(location: str) -> dict[str, str]:
    block = GetWeatherInformationBlock()
    return {
        name: value
        async for name, value in block.run(
            GetWeatherInformationBlock.Input(
                location=location, credentials=TEST_CREDENTIALS_INPUT
            ),
            credentials=TEST_CREDENTIALS,
        )
    }


@asynccontextmanager
async def _openweathermap(monkeypatch, respond):
    """Sends the block's real requests to a local server; yields the URLs it built."""
    requested: list[str] = []
    app = web.Application()
    app.router.add_get("/data/2.5/weather", respond)
    async with TestServer(app) as server:
        local = str(server.make_url("/"))

        class LocalWeatherRequests(Requests):
            def __init__(self):
                super().__init__(trusted_origins=[local], retry_max_attempts=1)

            async def get(self, url: str, *args, **kwargs):
                requested.append(url)
                sent = urlparse(url)
                local_url = f"{server.make_url(sent.path)}?{sent.query}"
                return await super().get(local_url, *args, **kwargs)

        monkeypatch.setattr(http, "Requests", LocalWeatherRequests)
        yield requested
