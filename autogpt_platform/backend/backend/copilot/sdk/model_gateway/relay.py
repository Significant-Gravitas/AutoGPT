"""Writing to the CLI: the SSE relay, per-request usage, and the JSON helpers."""

import asyncio
import time
from collections.abc import AsyncGenerator

from aiohttp import web

from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.sse import PING, encode_event
from backend.copilot.sdk.model_gateway.stream import Event, StreamItem
from backend.copilot.sdk.model_gateway.usage import (
    GatewayUsage,
    merge_stream_usage,
    usage_from_anthropic,
)

PING_INTERVAL_SECONDS = 15.0
SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


class UsageMeter:
    """One request's usage: the adapter's exact count when it gave one,
    otherwise what the events the CLI received said."""

    def __init__(self) -> None:
        self._start = GatewayUsage()
        self._delta = GatewayUsage()
        self._exact: GatewayUsage | None = None

    @property
    def total(self) -> GatewayUsage:
        if self._exact is not None:
            return self._exact
        return merge_stream_usage(self._start, self._delta)

    def observe(self, item: StreamItem) -> None:
        if item.usage is not None:
            self._exact = item.usage
        event = item.event
        kind = event.get("type")
        message = event.get("message")
        if kind == "message_start" and isinstance(message, dict):
            self._start = usage_from_anthropic(message.get("usage")) or self._start
        elif kind == "message_delta":
            self._delta = usage_from_anthropic(event.get("usage")) or self._delta

    def observe_message(self, message: Event) -> None:
        self._delta = usage_from_anthropic(message.get("usage")) or self._delta


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
        self._meter.observe(item)
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
