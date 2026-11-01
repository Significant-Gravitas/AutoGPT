"""
V2 External API - Request body size limit

ASGI middleware that bounds how much of a request body the v2 app reads.
FastAPI reads a JSON body whole before the route runs, and FastMCP reads a
JSON-RPC body whole before it dispatches the tool, so without a bound one
caller holding a few large requests open could exhaust a server's memory.

A body declared longer than its endpoint's cap is refused before anything
reads it. One sent in chunks is counted as the app reads it, and refused as
soon as it crosses the cap. This middleware sends that 413 itself and drops
whatever the app sends after it: FastAPI turns a failed body read into a 400,
and FastMCP into a 500.
"""

from typing import Optional

from starlette import status
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.util.settings import Settings

from .errors import error_response
from .global_rate_limit import app_path
from .marketplace import MEDIA_MAX_SIZE_MB

settings = Settings()

_MiB = 1024 * 1024
# Every request that is not an upload, an MCP tool call included.
MAX_BODY_BYTES = 10 * _MiB
# The multipart framing around an uploaded file.
_MULTIPART_OVERHEAD = _MiB


def max_body_bytes(path: str) -> int:
    """The most body a request to *path*, inside the v2 app, may send."""
    if path == "/files/upload":
        return settings.config.max_file_size_mb * _MiB + _MULTIPART_OVERHEAD
    if path == "/marketplace/submissions/media":
        return MEDIA_MAX_SIZE_MB * _MiB + _MULTIPART_OVERHEAD
    return MAX_BODY_BYTES


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = max_body_bytes(app_path(scope))
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await _too_large(limit)(scope, receive, send)
            return

        guard = _BodyGuard(scope, receive, send, limit)
        try:
            await self.app(scope, guard.receive, guard.send)
        except RequestBodyTooLarge:
            # Nothing caught the failed read on the way out. The 413 has gone
            # out already, unless the app had started its own response.
            pass


class RequestBodyTooLarge(Exception):
    """Raised into the app when the body it is reading crosses the cap."""


class _BodyGuard:
    """Counts the body as the app reads it, and answers 413 when it crosses
    the cap."""

    def __init__(self, scope: Scope, receive: Receive, send: Send, limit: int):
        self._scope = scope
        self._receive = receive
        self._send = send
        self._limit = limit
        self._received = 0
        self._response_started = False
        self._answered = False

    async def receive(self) -> Message:
        message = await self._receive()
        if message["type"] != "http.request":
            return message
        self._received += len(message.get("body", b""))
        if self._received <= self._limit:
            return message
        if not (self._response_started or self._answered):
            self._answered = True
            await _too_large(self._limit)(self._scope, self._receive, self._send)
        raise RequestBodyTooLarge()

    async def send(self, message: Message) -> None:
        if self._answered:
            # The app's own answer to the failed read; the 413 went out first.
            return
        if message["type"] == "http.response.start":
            self._response_started = True
        await self._send(message)


def _declared_length(scope: Scope) -> Optional[int]:
    value = Headers(scope=scope).get("content-length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _too_large(limit: int) -> Response:
    return error_response(
        status.HTTP_413_CONTENT_TOO_LARGE,
        f"The request body is larger than this endpoint accepts ({limit:,} bytes)",
        details={"limit_bytes": limit},
    )
