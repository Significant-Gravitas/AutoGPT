"""Bound how much of a request body the platform API reads.

FastAPI reads and parses a route's body before the route runs, so the bound
sits in front of the routes. Each route gets the limit of its kind: an upload
route's multipart body may carry a whole file, every other body is held to
`max_request_body_mb`.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import params
from fastapi.routing import APIRoute
from starlette import status
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.util.settings import Settings

settings = Settings()

MiB = 1024 * 1024
# The multipart framing around an uploaded file.
MULTIPART_OVERHEAD = MiB

# Set by a mounted app that bounds its own bodies, so its error shape answers.
_BOUNDED_INSIDE = "backend.request_body_bounded"

_Endpoint = TypeVar("_Endpoint", bound=Callable[..., Any])
_ENDPOINT_LIMITS: dict[Callable[..., Any], int] = {}


class RequestBodyLimitMiddleware:
    """Answers 413 when a body crosses the limit of the route it reached,
    without reading past that limit."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        guard = BodyGuard(scope, receive, send, route_body_limit, _too_large)
        try:
            await self.app(scope, guard.receive, guard.send)
        except RequestBodyTooLarge:
            # Nothing caught the failed read on the way out. The 413 has gone
            # out already, unless the app had started its own response.
            pass


def request_body_limit(max_bytes: int) -> Callable[[_Endpoint], _Endpoint]:
    """Give a route's body `max_bytes` instead of the default limit."""

    def mark(endpoint: _Endpoint) -> _Endpoint:
        _ENDPOINT_LIMITS[endpoint] = max_bytes
        return endpoint

    return mark


def route_body_limit(scope: Scope) -> int | None:
    """The most body a request may send to the route it was matched to; None
    when a mounted app bounds the body itself. Read once the route is known."""
    if scope.get(_BOUNDED_INSIDE):
        return None
    route = scope.get("route")
    if isinstance(route, APIRoute):
        own = _ENDPOINT_LIMITS.get(route.endpoint)
        if own is not None:
            return own
        if _takes_a_file(route) and _is_multipart(scope):
            return upload_body_limit()
    return settings.config.max_request_body_mb * MiB


def _takes_a_file(route: APIRoute) -> bool:
    return route.body_field is not None and isinstance(
        route.body_field.field_info, params.File
    )


def upload_body_limit() -> int:
    largest_file_mb = max(
        settings.config.upload_file_size_limit_mb, settings.config.max_file_size_mb
    )
    return largest_file_mb * MiB + MULTIPART_OVERHEAD


def mark_bounded_inside(scope: Scope) -> None:
    """Called by a mounted app's own body limit, which then answers instead."""
    scope[_BOUNDED_INSIDE] = True


class RequestBodyTooLarge(Exception):
    """Raised into the app when the body it is reading crosses the limit."""


class BodyGuard:
    """Counts the body as the app reads it, and answers 413 when it crosses
    the limit. The limit is resolved at the first read, once routing is done.

    The guard sends the 413 itself and drops whatever the app sends after it:
    FastAPI turns a failed body read into a 400, and FastMCP into a 500."""

    def __init__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        limit: Callable[[Scope], int | None],
        too_large: Callable[[int], Response],
    ):
        self._scope = scope
        self._receive = receive
        self._send = send
        self._resolve_limit = limit
        self._too_large = too_large
        self._limit: int | None = None
        self._resolved = False
        self._received = 0
        self._refused = False
        self._response_started = False
        self._answered = False

    async def receive(self) -> Message:
        if not self._resolved:
            self._resolved = True
            self._limit = self._resolve_limit(self._scope)
            declared = declared_length(self._scope)
            if self._limit is not None and (declared or 0) > self._limit:
                await self._refuse()
        if self._refused:
            raise RequestBodyTooLarge()

        message = await self._receive()
        if message["type"] != "http.request" or self._limit is None:
            return message
        self._received += len(message.get("body", b""))
        if self._received > self._limit:
            await self._refuse()
            raise RequestBodyTooLarge()
        return message

    async def send(self, message: Message) -> None:
        if self._answered:
            # The app's own answer to the failed read; the 413 went out first.
            return
        if message["type"] == "http.response.start":
            self._response_started = True
        await self._send(message)

    async def _refuse(self) -> None:
        self._refused = True
        if self._response_started or self._answered:
            return
        assert self._limit is not None
        self._answered = True
        await self._too_large(self._limit)(self._scope, self._receive, self._send)


def declared_length(scope: Scope) -> int | None:
    value = Headers(scope=scope).get("content-length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _is_multipart(scope: Scope) -> bool:
    content_type = Headers(scope=scope).get("content-type", "")
    return content_type.lower().startswith("multipart/form-data")


def _too_large(limit: int) -> Response:
    return JSONResponse(
        {
            "detail": f"The request body is larger than this endpoint accepts ({limit:,} bytes)"
        },
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
    )
