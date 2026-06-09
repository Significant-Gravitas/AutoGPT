"""
V2 External API - Global Rate Limit Middleware

ASGI middleware that enforces a per-user request cap across all v2 endpoints.
Extracts the user identity from the API key or Bearer token header and checks
a Redis-backed rate limiter. If the user exceeds the cap, returns 429 before
the request reaches the route handler.

On auth-resolution failure or Redis errors the request passes through — the
endpoint's own auth dependency handles 401, and the rate limiter fails open.
"""

import json
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

from backend.api.utils.rate_limit import RateLimiter
from backend.data.auth.api_key import validate_api_key
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    validate_access_token,
)

logger = logging.getLogger(__name__)

_limiter = RateLimiter("v2:global", max_requests=200, window_seconds=60)


async def _resolve_user_id(scope: Scope) -> str | None:
    headers = dict(scope.get("headers", []))
    api_key = headers.get(b"x-api-key", b"").decode()
    if api_key:
        info = await validate_api_key(api_key)
        return info.user_id if info else None

    auth_header = headers.get(b"authorization", b"").decode()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        try:
            token_info, _ = await validate_access_token(token)
            return token_info.user_id
        except (InvalidClientError, InvalidTokenError):
            return None

    return None


class GlobalRateLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        user_id = await _resolve_user_id(scope)
        if user_id:
            from fastapi import HTTPException

            try:
                await _limiter.check(user_id)
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
