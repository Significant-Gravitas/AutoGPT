"""Writing to the CLI: the SSE relay, per-request usage, and the JSON helpers."""

import asyncio
import time
from collections.abc import AsyncGenerator

from aiohttp import web

from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.sse import PING, encode_event
from backend.copilot.sdk.model_gateway.stream import Event, StreamItem
from backend.copilot.sdk.model_gateway.usage import UsageMeter

PING_INTERVAL_SECONDS = 15.0
SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


async def relay(
    response: web.StreamResponse,
    first: StreamItem,
    items: AsyncGenerator[StreamItem, None],
    meter: UsageMeter,
) -> None:
    """Write every item, with a ping whenever the upstream goes quiet.

    The upstream is read in this task only; a side task writes the pings.
    """
    writer = _SseWriter(response, meter)
    pinger = asyncio.create_task(writer.ping_while_quiet())
    try:
        await writer.write(first)
        async for item in items:
            await writer.write(item)
    finally:
        pinger.cancel()
        await asyncio.gather(pinger, return_exceptions=True)


class _SseWriter:
    """Serialises item and ping writes to one response."""

    def __init__(self, response: web.StreamResponse, meter: UsageMeter) -> None:
        self._response = response
        self._meter = meter
        self._lock = asyncio.Lock()
        self._last_write = time.monotonic()

    async def write(self, item: StreamItem) -> None:
        self._meter.observe(item.event, item.usage)
        async with self._lock:
            await self._response.write(item.raw or encode_event(item.event))
            self._last_write = time.monotonic()

    async def ping_while_quiet(self) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            async with self._lock:
                if time.monotonic() - self._last_write >= PING_INTERVAL_SECONDS:
                    await self._response.write(encode_event(PING))
                    self._last_write = time.monotonic()


async def json_body(request: web.Request) -> Event | web.Response:
    """The request's JSON object, or the 400 to answer with."""
    try:
        body = await request.json()
    except ValueError:
        return error_response(
            GatewayError(400, "invalid_request_error", "The request body must be JSON")
        )
    if not isinstance(body, dict):
        return error_response(
            GatewayError(400, "invalid_request_error", "The body must be an object")
        )
    return body


def error_response(error: GatewayError) -> web.Response:
    return web.json_response(error.body(), status=error.status, headers=error.headers)


async def write_quietly(response: web.StreamResponse, data: bytes) -> None:
    """Write to a CLI that may already have hung up."""
    try:
        await response.write(data)
    except ConnectionError:
        return


async def end_quietly(response: web.StreamResponse) -> None:
    try:
        await response.write_eof()
    except ConnectionError:
        return
