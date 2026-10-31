import re

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.monitoring.instrumentation import record_legacy_api_path_hit

INTERNAL_API_PREFIX = "/api/internal"
_PUBLIC_API_VERSION = re.compile(r"v\d+")


class LegacyAPIPathMiddleware:
    """Serves ``/api/<segment>/…`` as ``/api/internal/<segment>/…`` for the internal
    API's own segments: third parties, emails, stored chats and open tabs hold them."""

    def __init__(self, app: ASGIApp, segments: frozenset[str]):
        self.app = app
        # `/api/v<n>` is the public API's, whatever an internal route is called.
        self.segments = frozenset(
            s for s in segments if not _PUBLIC_API_VERSION.fullmatch(s)
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = self._rewrite(scope)
        await self.app(scope, receive, send)

    def _rewrite(self, scope: Scope) -> Scope:
        path: str = scope["path"]
        if not path.startswith("/api/"):
            return scope
        segment = path[len("/api/") :].split("/", 1)[0]
        if segment not in self.segments:
            return scope
        record_legacy_api_path_hit(segment)
        rewritten = {**scope, "path": INTERNAL_API_PREFIX + path[len("/api") :]}
        raw_path: bytes | None = scope.get("raw_path")
        if raw_path is not None and raw_path.startswith(b"/api/"):
            rewritten["raw_path"] = (
                INTERNAL_API_PREFIX.encode() + raw_path[len(b"/api") :]
            )
        return rewritten


def internal_api_segments(app: FastAPI) -> frozenset[str]:
    """The first path segment under ``/api/internal`` of every route of `app`."""
    prefix = INTERNAL_API_PREFIX + "/"
    return frozenset(
        route.path[len(prefix) :].split("/", 1)[0]
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith(prefix)
    )
