"""
V2 External API - Global Rate Limit Middleware

ASGI middleware that enforces per-user and per-IP request caps across all v2
endpoints. Authenticated users get 200 req/min keyed by user ID; unauthenticated
sessions get 5 req/min keyed by client IP.

Identifies the user through the auth middleware's `resolve_request_auth`.
Verifying an API key costs a Scrypt hash, which the per-user cap can't bound:
it can only count a request once the hash is done. So before any hashing,
API keys are counted per client IP and key head (the part of the key the
lookup matches on): a head presented too often, or failing too often, is
refused unhashed until the window rolls over, as is an address failing too
often across heads. A head is shared by few keys, so one client's bad key or
flood doesn't lock out the other keys behind the same address.

Every response carries the caller's `X-RateLimit-*` position, and a 429 adds
`Retry-After`, so a client can back off on the numbers instead of guessing.

On auth-resolution failure or Redis errors the request passes through — the
endpoint's own auth dependency handles 401, and the rate limiter fails open.
"""

import contextlib
import logging
from typing import Optional

from autogpt_libs.api_key.keysmith import APIKeySmith
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.api.external.middleware import resolve_request_auth
from backend.api.utils.rate_limit import RateLimiter, RateLimitState
from backend.data.auth.oauth import is_access_token
from backend.util.settings import Settings

from .errors import error_response

logger = logging.getLogger(__name__)
settings = Settings()

_authenticated_limiter = RateLimiter("v2:global", max_requests=200, window_seconds=60)
_anonymous_limiter = RateLimiter("v2:global:anon", max_requests=5, window_seconds=60)
# Above the per-user cap, so a client within its own limit never meets it.
_presented_key_limiter = RateLimiter(
    "v2:global:key-presented", max_requests=300, window_seconds=60
)
_failed_key_limiter = RateLimiter(
    "v2:global:key-failures", max_requests=30, window_seconds=60
)
# Failures spread over many heads, each of which may match a key to hash.
_failed_auth_limiter = RateLimiter(
    "v2:global:auth-failures", max_requests=300, window_seconds=60
)
# The docs tell agents to read the spec before they have a key; on the 5/min
# bucket those reads would use up what the first real calls need.
_anonymous_docs_limiter = RateLimiter(
    "v2:global:anon-docs", max_requests=60, window_seconds=60
)
_DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


class GlobalRateLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        # The first of a repeated header, as the route's own dependency reads
        # it, so both charge and authenticate the same credential.
        request_headers = Headers(scope=scope)

        api_key = request_headers.get("x-api-key") or None
        auth_header = request_headers.get("authorization", "")
        bearer = None
        if auth_header.lower().startswith("bearer "):
            bearer = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=auth_header[7:]
            )

        ip = client_ip(scope, headers)
        key_bucket = _key_bucket(ip, api_key, bearer)
        if key_bucket is not None and (
            refusal := await _refuse_before_hashing(ip, key_bucket)
        ):
            await refusal(scope, receive, send)
            return

        try:
            auth = await resolve_request_auth(scope, api_key=api_key, bearer=bearer)
        except HTTPException as rejection:
            auth = None
            if rejection.status_code == 401 and key_bucket is not None:
                with contextlib.suppress(HTTPException):
                    await _failed_key_limiter.check(key_bucket)
                with contextlib.suppress(HTTPException):
                    await _failed_auth_limiter.check(ip)
        except Exception as exc:
            # Fail open on anything the auth backend throws that is not a
            # rejection; the route's own dependency will answer 401 or 500.
            logger.warning(f"Rate-limit auth resolution failed: {exc}")
            auth = None

        try:
            if auth:
                state = await _authenticated_limiter.check(auth.user_id)
            elif _app_path(scope) in _DOCS_PATHS:
                state = await _anonymous_docs_limiter.check(ip)
            else:
                state = await _anonymous_limiter.check(ip)
        except HTTPException as exc:
            # The middleware sits outside the app, so the v2 exception handlers
            # never see this — build the same envelope by hand.
            response = error_response(
                exc.status_code, str(exc.detail), headers=exc.headers
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, _with_rate_limit_headers(send, state))


def _key_bucket(
    ip: str, api_key: Optional[str], bearer: Optional[HTTPAuthorizationCredentials]
) -> Optional[str]:
    """The pre-verification bucket of a credential that costs a hash, or None.

    API keys are matched on their head and then hashed. A bearer value in the
    OAuth access-token format is only ever looked up by its digest (see
    `resolve_auth_info`), so it costs no hash and gets no bucket; the same
    value sent as `X-API-Key` is tried as a key, so it does.
    """
    if api_key is not None:
        credential = api_key
    elif bearer is not None and not is_access_token(bearer.credentials):
        credential = bearer.credentials
    else:
        return None
    if not credential.startswith(APIKeySmith.PREFIX):
        return None
    return f"{ip}:{credential[: APIKeySmith.HEAD_LENGTH]}"


async def _refuse_before_hashing(ip: str, key_bucket: str) -> Optional[Response]:
    """A 429 for a key that may not be hashed now, or None to go on.

    Counts the presentation itself, so a valid key flooding the API is refused
    unhashed past the cap, as is a head or an address that keeps failing.
    """
    if await _failed_key_limiter.exhausted(key_bucket) or (
        await _failed_auth_limiter.exhausted(ip)
    ):
        return error_response(
            429,
            "Too many failed authentication attempts. Try again shortly.",
            headers={"Retry-After": str(_failed_key_limiter.window_seconds)},
        )
    try:
        await _presented_key_limiter.check(key_bucket)
    except HTTPException as exc:
        return error_response(exc.status_code, str(exc.detail), headers=exc.headers)
    return None


def _app_path(scope: Scope) -> str:
    """The request path inside this app, past the prefix it is mounted at."""
    path: str = scope.get("path", "")
    root: str = scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def client_ip(scope: Scope, headers: dict[bytes, bytes]) -> str:
    """The caller's address, trusting only the proxies in front of us.

    Our proxies append `trusted_proxy_count` entries, so the client is that
    many from the right; anything further left the caller wrote itself and
    could use to spread its requests over an unlimited number of buckets.
    It counts entries, not proxies: Google's load balancer appends two.
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
            headers = message.setdefault("headers", [])
            # An endpoint with its own, narrower limiter has already set these.
            present = {name.lower() for name, _ in headers}
            headers.extend(h for h in encoded if h[0] not in present)
        await send(message)

    return send_with_headers
