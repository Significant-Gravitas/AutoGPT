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
from backend.util.request import Requests


@pytest.mark.asyncio
async def test_weather_request_carries_the_real_key_over_https(monkeypatch):
    key = TEST_CREDENTIALS.api_key.get_secret_value()
    requested: list[str] = []

    # Answers as OpenWeatherMap does: 401 unless appid is a real key.
    async def respond(incoming: web.Request):
        if incoming.query.get("appid") != key:
            return web.json_response(
                {"cod": 401, "message": "Invalid API key"}, status=401
            )
        return web.json_response(
            {
                "main": {"temp": 12.5, "humidity": 81},
                "weather": [{"description": "light rain"}],
            }
        )

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
        block = GetWeatherInformationBlock()
        outputs = {
            name: value
            async for name, value in block.run(
                GetWeatherInformationBlock.Input(
                    location="London", credentials=TEST_CREDENTIALS_INPUT
                ),
                credentials=TEST_CREDENTIALS,
            )
        }

    sent = urlparse(requested[0])
    assert (sent.scheme, sent.hostname) == ("https", "api.openweathermap.org")
    assert outputs == {
        "temperature": "12.5",
        "humidity": "81",
        "condition": "light rain",
    }
