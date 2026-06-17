"""
V2 External API - Global Rate Limit Middleware

ASGI middleware that enforces per-user and per-IP request caps across all v2
endpoints. Authenticated users get 200 req/min keyed by user ID; unauthenticated
sessions get 5 req/min keyed by client IP.

Reuses `resolve_auth_info` from the auth middleware to identify the user.
On auth-resolution failure or Redis errors the request passes through — the
endpoint's own auth dependency handles 401, and the rate limiter fails open.
"""

import json
import logging

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.api.external.middleware import resolve_auth_info
from backend.api.utils.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

_authenticated_limiter = RateLimiter("v2:global", max_requests=200, window_seconds=60)
_anonymous_limiter = RateLimiter("v2:global:anon", max_requests=5, window_seconds=60)


class GlobalRateLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))

        api_key = headers.get(b"x-api-key", b"").decode() or None
        auth_header = headers.get(b"authorization", b"").decode()
        bearer = None
        if auth_header.lower().startswith("bearer "):
            bearer = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=auth_header[7:]
            )

        try:
            auth = await resolve_auth_info(api_key=api_key, bearer=bearer)
        except HTTPException:
            auth = None

        try:
            if auth:
                await _authenticated_limiter.check(auth.user_id)
            else:
                ip = (
                    headers.get(b"x-forwarded-for", b"").decode().split(",")[0].strip()
                    or (scope.get("client") or ("unknown",))[0]
                )
                await _anonymous_limiter.check(ip)
        except HTTPException as exc:
            body = json.dumps({"detail": exc.detail}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": exc.status_code,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(body)).encode()],
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
