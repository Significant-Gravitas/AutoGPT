from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from backend.blocks.anysearch import _api

REQUEST_ID = "5a3f8c27-1e64-4b90-a752-6d9e2f41c083"
HEADER_REQUEST_ID = "2e5a8c74-1d39-4f62-b087-9c6e3a51d428"
QUOTA_RESPONSE = {
    "code": -1,
    "message": "Your account and API key have been automatically generated. "
    "Use the API key below to continue.\n"
    "username=<USERNAME>\npassword=<PASSWORD>\napi_key=<API_KEY>",
    "request_id": REQUEST_ID,
}


@asynccontextmanager
async def anysearch_server(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[list[str]]:
    calls: list[str] = []

    async def route(request: web.Request) -> web.StreamResponse:
        calls.append(request.path)
        return await handler(request)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", route)
    async with TestServer(app, host="127.0.0.1") as server:
        monkeypatch.setattr(_api, "ANYSEARCH_API_URL", str(server.make_url("")))
        yield calls


def assert_no_credentials(text: str) -> None:
    for value in ("<USERNAME>", "<PASSWORD>", "<API_KEY>"):
        assert value not in text
