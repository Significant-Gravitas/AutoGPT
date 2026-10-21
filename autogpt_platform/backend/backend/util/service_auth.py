"""The shared key that authenticates calls between internal services.

A method an ``AppService`` exposes runs for anything that can reach the
service's port, and many of them read or change any user's data. So every
request to an RPC service must carry ``Authorization: Bearer <key>``, which
``get_service_client`` adds to every call, and a service refuses any other
request before reading its body. Only the health checks and ``/metrics`` are
open, for probes and the Prometheus scraper; neither returns user data. The
header is ``Authorization`` because Sentry and the event scrubber in
``backend.util.metrics`` already keep its value out of error reports.

The key is ``INTERNAL_API_KEY`` when set, and otherwise derived from
``ENCRYPTION_KEY``, which every backend service of a deployment already
shares: a deployment needs nothing new to enforce it. A derived key changes
with ``ENCRYPTION_KEY``, so restart every service together when rotating that,
or set ``INTERNAL_API_KEY`` first.

``INTERNAL_API_KEY_OPTIONAL`` is for rolling the key out to a deployment whose
older instances cannot send it yet: a call that carries no key is accepted
and counted, while a call with a wrong key is still refused.
"""

import base64
import hashlib
import hmac
import logging
import re

from fastapi import FastAPI
from prometheus_client import Counter
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

OPEN_PATHS = frozenset({"/health_check", "/health_check_async", "/metrics"})

_VALID_KEY = re.compile(r"[A-Za-z0-9._~-]{32,}={0,2}")
_DERIVATION_LABEL = b"autogpt internal service key v1"
_REFUSAL = b'{"detail":"Invalid internal service key"}'

calls_without_key = Counter(
    "autogpt_internal_rpc_calls_without_key_total",
    "Internal RPC calls accepted without a key while INTERNAL_API_KEY_OPTIONAL "
    "is on. Turn the setting off once this stops growing.",
    ["service"],
)


def internal_api_key(settings: Settings | None = None) -> str:
    """The key internal calls carry: ``INTERNAL_API_KEY``, else one derived
    from ``ENCRYPTION_KEY``, else ``""`` when neither is set.

    Raises ``ValueError`` on a malformed ``INTERNAL_API_KEY``; every process
    checks it at start (``AppProcess.start``).
    """
    secrets = (settings or Settings()).secrets
    if secrets.internal_api_key:
        if not _VALID_KEY.fullmatch(secrets.internal_api_key):
            raise ValueError(
                "INTERNAL_API_KEY must be at least 32 URL-safe characters: "
                "letters, digits, '-', '_', '.', '~' (and '=' padding)."
            )
        return secrets.internal_api_key
    if not secrets.encryption_key:
        return ""
    digest = hmac.new(
        secrets.encryption_key.encode(), _DERIVATION_LABEL, hashlib.sha256
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def internal_auth_headers(settings: Settings | None = None) -> dict[str, str]:
    """The headers an RPC client sends with every call."""
    key = internal_api_key(settings)
    return {"Authorization": f"Bearer {key}"} if key else {}


def require_internal_key(
    app: FastAPI, service_name: str, settings: Settings | None = None
) -> None:
    """Make *app* refuse every request outside ``OPEN_PATHS`` that does not
    carry the key."""
    settings = settings or Settings()
    key = internal_api_key(settings)
    if not key:
        logger.warning(
            f"[SECURITY] [{service_name}] No internal service key: set "
            "ENCRYPTION_KEY (every service needs it) or INTERNAL_API_KEY. Until "
            "then anything that can reach this port can call its methods."
        )
        return
    optional = settings.config.internal_api_key_optional
    if optional:
        logger.warning(
            f"[SECURITY] [{service_name}] INTERNAL_API_KEY_OPTIONAL is on: calls "
            "without the key are accepted. Turn it off once no instance started "
            "before the key was sent is still running "
            "(autogpt_internal_rpc_calls_without_key_total stops growing)."
        )
    app.add_middleware(
        InternalKeyMiddleware, service_name=service_name, key=key, optional=optional
    )


class InternalKeyMiddleware:
    """Refuses a request outside ``OPEN_PATHS`` without the key, before its
    body is read: the bearer token is compared in constant time, and only an
    absent one passes while the key is optional."""

    def __init__(self, app: ASGIApp, service_name: str, key: str, optional: bool):
        self.app = app
        self.service_name = service_name
        self.expected = key.encode()
        self.optional = optional

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        supplied = _bearer_token(scope)
        if supplied is not None and hmac.compare_digest(supplied, self.expected):
            await self.app(scope, receive, send)
            return
        if supplied is None and self.optional:
            calls_without_key.labels(self.service_name).inc()
            await self.app(scope, receive, send)
            return
        caller = (scope.get("client") or ("unknown",))[0]
        logger.warning(
            f"[{self.service_name}] Refused {scope['path']!r} from {caller}: "
            "missing or wrong internal service key"
        )
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_REFUSAL)).encode()),
                    (b"www-authenticate", b"Bearer"),
                    # Close instead of taking in the rest of a refused body.
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _REFUSAL})


def _bearer_token(scope: Scope) -> bytes | None:
    """The first ``Authorization`` header's bearer token: ``None`` when there
    is no such header, ``b""`` when it holds anything but a bearer token."""
    for name, value in scope["headers"]:
        if name == b"authorization":
            scheme, _, token = value.partition(b" ")
            return token if scheme.lower() == b"bearer" else b""
    return None
