"""The shared key that authenticates calls between internal services.

A method an ``AppService`` exposes runs for anything that can reach the
service's port, and many of them read or change any user's data. So every
request to an RPC service must carry ``Authorization: Bearer <key>``, which
``get_service_client`` adds to every call, and a service refuses any other
request before reading its body. Only the health checks and ``/metrics`` are
open, for probes and the Prometheus scraper; neither returns user data. The
header is ``Authorization`` because Sentry and the event scrubber in
``backend.util.metrics`` already keep its value out of error reports.

The key is ``INTERNAL_API_KEY``, which setup generates along with the other
secrets. An install set up before it existed has none, so there it is derived
from ``ENCRYPTION_KEY``, which every backend service of a deployment already
shares. A derived key changes with ``ENCRYPTION_KEY``, so restart every service
together when rotating that. A service with neither refuses every call.
"""

import base64
import hashlib
import hmac
import logging
import re

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.util.secrets_guard import RETIRED_HINT
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

OPEN_PATHS = frozenset({"/health_check", "/health_check_async", "/metrics"})

_VALID_KEY = re.compile(r"[A-Za-z0-9._~-]{32,}={0,2}")
_DERIVATION_LABEL = b"autogpt internal service key v1"
_REFUSAL = b'{"detail":"Invalid internal service key"}'


def internal_api_key(settings: Settings | None = None) -> str:
    """The key internal calls carry: ``INTERNAL_API_KEY``, else one derived
    from ``ENCRYPTION_KEY``, else ``""`` when neither is set.

    Raises ``ValueError`` on a malformed ``INTERNAL_API_KEY``, and on an
    ``ENCRYPTION_KEY`` that is not a Fernet key when the key is derived from
    it: a guessable one would make the internal key guessable too. Every
    process checks this at start (``AppProcess.start``).
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
    source = _fernet_key_bytes(secrets.encryption_key)
    if source is None:
        raise ValueError(
            "ENCRYPTION_KEY is not a Fernet key (32 random bytes, URL-safe "
            "base64), and the internal service key is derived from it. "
            + RETIRED_HINT.format(name="ENCRYPTION_KEY")
        )
    digest = hmac.new(source, _DERIVATION_LABEL, hashlib.sha256).digest()
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
    key = internal_api_key(settings)
    if not key:
        logger.error(
            f"[SECURITY] [{service_name}] No internal service key: set "
            "ENCRYPTION_KEY (every service needs it) or INTERNAL_API_KEY. Until "
            "then every call but the health checks is refused."
        )
    app.add_middleware(InternalKeyMiddleware, service_name=service_name, key=key)


class InternalKeyMiddleware:
    """Refuses a request outside ``OPEN_PATHS`` without the key, before its
    body is read; the bearer token is compared in constant time. Without a
    key it refuses all of them."""

    def __init__(self, app: ASGIApp, service_name: str, key: str):
        self.app = app
        self.service_name = service_name
        self.expected = key.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        # Without a key nothing matches, not even an empty token.
        if self.expected and hmac.compare_digest(_bearer_token(scope), self.expected):
            await self.app(scope, receive, send)
            return
        await self._refuse(scope, send)

    async def _refuse(self, scope: Scope, send: Send) -> None:
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


def _bearer_token(scope: Scope) -> bytes:
    """The first ``Authorization`` header's bearer token, or ``b""`` when
    there is no such header or it holds anything but a bearer token."""
    for name, value in scope["headers"]:
        if name == b"authorization":
            scheme, _, token = value.partition(b" ")
            return token if scheme.lower() == b"bearer" else b""
    return b""


def _fernet_key_bytes(value: str) -> bytes | None:
    """The 32 bytes ``Fernet``, and so ``JSONCryptor``, reads from *value*, or
    ``None`` where it refuses it. Deriving from these rather than the text
    gives every encoding of one key that Fernet accepts the same internal key."""
    try:
        key = base64.urlsafe_b64decode(value.encode())
    except ValueError:  # binascii.Error is a ValueError
        return None
    return key if len(key) == 32 else None
