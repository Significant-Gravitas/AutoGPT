"""
V2 External API - Request body size limit

ASGI middleware that bounds how much of a request body the v2 app reads.
FastAPI reads a JSON body whole before the route runs, and FastMCP reads a
JSON-RPC body whole before it dispatches the tool, so without a bound one
caller holding a few large requests open could exhaust a server's memory.

A body declared longer than its endpoint's cap is refused before anything
reads it. One sent in chunks is counted as the app reads it, and refused as
soon as it crosses the cap.
"""

from starlette import status
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.api.middleware.body_limit import (
    BodyGuard,
    RequestBodyTooLarge,
    declared_length,
    mark_bounded_inside,
)
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

        mark_bounded_inside(scope)
        limit = max_body_bytes(app_path(scope))
        declared = declared_length(scope)
        if declared is not None and declared > limit:
            await _too_large(limit)(scope, receive, send)
            return

        guard = BodyGuard(scope, receive, send, lambda _: limit, _too_large)
        try:
            await self.app(scope, guard.receive, guard.send)
        except RequestBodyTooLarge:
            # Nothing caught the failed read on the way out. The 413 has gone
            # out already, unless the app had started its own response.
            pass


def _too_large(limit: int) -> Response:
    return error_response(
        status.HTTP_413_CONTENT_TOO_LARGE,
        f"The request body is larger than this endpoint accepts ({limit:,} bytes)",
        details={"limit_bytes": limit},
    )
