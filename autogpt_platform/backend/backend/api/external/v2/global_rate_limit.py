"""
V2 External API - Global Rate Limit Middleware

ASGI middleware that enforces per-user and per-IP request caps across all v2
endpoints. Authenticated users get 200 req/min keyed by user ID; unauthenticated
sessions get 5 req/min keyed by client IP.

Reuses `resolve_auth_info` from the auth middleware to identify the user, and
hands the result to the route's dependency through the request scope so the
credential is verified once per request.

Every response carries the caller's `X-RateLimit-*` position, and a 429 adds
`Retry-After`, so a client can back off on the numbers instead of guessing.

On auth-resolution failure or Redis errors the request passes through — the
endpoint's own auth dependency handles 401, and the rate limiter fails open.
"""

import logging
from typing import Optional

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.api.external.middleware import resolve_auth_info
from backend.api.utils.rate_limit import RateLimiter, RateLimitState
from backend.util.settings import Settings

from .errors import error_response

logger = logging.getLogger(__name__)
settings = Settings()

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
        except Exception as exc:
            # Fail open on anything the auth backend throws that is not a
            # rejection; the route's own dependency will answer 401 or 500.
            logger.warning(f"Rate-limit auth resolution failed: {exc}")
            auth = None

        # The route's auth dependency reads this instead of verifying the same
        # credential a second time; an API key costs a Scrypt hash per check.
        if auth:
            scope.setdefault("state", {})["v2_auth"] = auth

        try:
            if auth:
                state = await _authenticated_limiter.check(auth.user_id)
            else:
                state = await _anonymous_limiter.check(client_ip(scope, headers))
        except HTTPException as exc:
            # The middleware sits outside the app, so the v2 exception handlers
            # never see this — build the same envelope by hand.
            response = error_response(
                exc.status_code, str(exc.detail), headers=exc.headers
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, _with_rate_limit_headers(send, state))


def client_ip(scope: Scope, headers: dict[bytes, bytes]) -> str:
    """The caller's address, trusting only the proxies in front of us.

    Each proxy appends the address it received the connection from, so with
    `trusted_proxy_count` proxies the client is that many entries from the
    right; anything further left the caller wrote itself and could use to
    spread its requests over an unlimited number of buckets.
    """
    peer = (scope.get("client") or ("unknown",))[0]
    hops = settings.config.trusted_proxy_count
    if hops < 1:
        return peer

    forwarded = [
        value.strip()
        for value in headers.get(b"x-forwarded-for", b"").decode().split(",")
        if value.strip()
    ]
    return forwarded[-hops] if len(forwarded) >= hops else peer


def _with_rate_limit_headers(send: Send, state: Optional[RateLimitState]) -> Send:
    """Attach the caller's window position to the response headers."""
    if state is None:
        return send

    encoded = [
        (name.lower().encode(), value.encode())
        for name, value in state.headers().items()
    ]

    async def send_with_headers(message: Message) -> None:
        if message["type"] == "http.response.start":
            message.setdefault("headers", []).extend(encoded)
        await send(message)

    return send_with_headers
