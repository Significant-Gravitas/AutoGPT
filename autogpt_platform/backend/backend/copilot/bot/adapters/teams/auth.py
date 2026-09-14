"""Inbound Bot Connector authentication and outbound token minting.

Teams differs from every other adapter we run: Slack and Telegram authenticate
a webhook by HMAC-ing the raw body against a shared secret, so one cheap
synchronous comparison proves both *who sent it* and *that the bytes are
untampered*. The Bot Connector instead sends a bearer JWT signed by Microsoft,
which authenticates **the sender only — it does not sign the payload**.

Two consequences drive the shape of this module:

1. Validation needs network I/O (fetching and rotating Microsoft's signing
   keys), so it cannot be expressed through the shared synchronous
   ``read_verified_webhook_body`` verifier. The adapter still reads the raw
   body first and refuses to parse it until the token checks out, preserving
   the same verify-before-parse ordering.
2. Because the token does not cover the body, a valid token may be replayed
   with *any* payload for its ~1h lifetime. Three layers narrow that gap: the
   ``serviceUrl`` claim binding below, the outbound host allowlist, and the
   adapter's per-activity-id dedupe.
"""

import asyncio
import ipaddress
import logging
import os
import socket
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import jwt

from . import config

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")

# Docker Desktop's DNS alias for the host's loopback interface.
_CONTAINER_HOST_ALIAS = "host.docker.internal"

# Microsoft rotates Connector signing keys; the discovery document is cheap and
# highly cacheable, so refresh daily rather than per request.
_KEY_CACHE_TTL_SECONDS = 24 * 60 * 60

# An unknown ``kid`` may mean a legitimate rotation — or an attacker spraying
# random values to force outbound fetches. Refresh at most this often so a
# spray can't turn every replica into a load generator against Microsoft.
_UNKNOWN_KID_REFRESH_INTERVAL_SECONDS = 300

# Microsoft's documented clock-skew allowance for Connector tokens.
_CLOCK_SKEW_SECONDS = 300

_HTTP_TIMEOUT_SECONDS = 10.0


class TeamsAuthError(Exception):
    """An inbound request failed Bot Connector authentication.

    ``status_code`` distinguishes a failed identity check (401) from a caller
    whose identity is proven but who is not endorsed for this channel (403),
    which Microsoft specifies as the correct response to an endorsement
    mismatch.
    """

    def __init__(self, message: str, *, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


class ConnectorTokenValidator:
    """Validates inbound JWTs issued by the Bot Connector.

    Holds the signing-key cache. One instance per adapter; safe to share
    across concurrent requests.
    """

    def __init__(self) -> None:
        self._keys: dict[str, dict[str, Any]] = {}
        self._fetched_at = 0.0
        # -inf, not 0.0: monotonic() is process uptime, so a zero start would
        # throttle the first forced refresh for the process's first 5 minutes.
        self._last_forced_refresh = float("-inf")
        self._lock = asyncio.Lock()

    async def validate(self, authorization: str | None) -> dict[str, Any]:
        """Return the token's claims, or raise :class:`TeamsAuthError`.

        This proves only that the Bot Connector sent the request. The caller
        MUST additionally bind the token to the request body via
        :func:`verify_service_url` — the token does not sign the payload.
        """
        token = _bearer_token(authorization)

        # Pin the algorithm from the UNVERIFIED header before touching key
        # material: never let a token nominate its own algorithm family, which
        # is how "alg: none" and HMAC-with-public-key forgeries work.
        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError as e:
            raise TeamsAuthError(f"malformed token: {e}")
        if header.get("alg") != "RS256":
            raise TeamsAuthError(f"unexpected signing algorithm {header.get('alg')!r}")

        kid = header.get("kid")
        if not kid:
            raise TeamsAuthError("token header carries no key id")

        raw_key = await self._key_for(str(kid))

        try:
            claims = jwt.decode(
                token,
                key=jwt.PyJWK(raw_key).key,
                algorithms=["RS256"],
                audience=config.get_app_id(),
                issuer=config.TOKEN_ISSUER,
                leeway=_CLOCK_SKEW_SECONDS,
            )
        except jwt.InvalidTokenError as e:
            raise TeamsAuthError(f"token rejected: {e}")

        _check_endorsements(raw_key)
        return claims

    async def _key_for(self, kid: str) -> dict[str, Any]:
        await self._ensure_keys()
        key = self._keys.get(kid)
        if key is not None:
            return key
        # Unknown kid: possibly a rotation we haven't picked up yet. One
        # throttled refresh, then give up rather than fetching per request.
        if time.monotonic() - self._last_forced_refresh >= (
            _UNKNOWN_KID_REFRESH_INTERVAL_SECONDS
        ):
            await self._ensure_keys(force=True)
            key = self._keys.get(kid)
        if key is None:
            raise TeamsAuthError(f"no signing key matches key id {kid!r}")
        return key

    async def _ensure_keys(self, *, force: bool = False) -> None:
        fresh = time.monotonic() - self._fetched_at < _KEY_CACHE_TTL_SECONDS
        if self._keys and fresh and not force:
            return
        async with self._lock:
            # Re-check under the lock so concurrent requests share one fetch.
            fresh = time.monotonic() - self._fetched_at < _KEY_CACHE_TTL_SECONDS
            if self._keys and fresh and not force:
                return
            if force:
                self._last_forced_refresh = time.monotonic()
            keys = await _fetch_signing_keys()
            if not keys:
                raise TeamsAuthError("no Bot Connector signing keys available")
            self._keys = keys
            self._fetched_at = time.monotonic()


async def _fetch_signing_keys() -> dict[str, dict[str, Any]]:
    """Resolve the discovery document, then its JWKS, keyed by ``kid``.

    Raw JWK dicts are kept (rather than parsed key objects) because Microsoft
    attaches a non-standard ``endorsements`` list to each key that the JWK
    parsers drop.
    """
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
        try:
            metadata = (await client.get(config.OPENID_METADATA_URL)).raise_for_status()
            jwks_uri = metadata.json()["jwks_uri"]
            jwks = (await client.get(jwks_uri)).raise_for_status().json()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise TeamsAuthError(f"could not load Bot Connector signing keys: {e}")
    return {k["kid"]: k for k in jwks.get("keys", []) if k.get("kid")}


def _check_endorsements(raw_key: dict[str, Any]) -> None:
    """Log — but do not yet enforce — the channel endorsement on the key.

    Microsoft documents rejecting an un-endorsed key with 403. Enforcing that
    on day one is a live-fire risk: if the key that actually signs ``msteams``
    traffic turns out not to carry the endorsement, hard enforcement would
    reject *all* inbound traffic. Observe real tokens first, then enforce.
    """
    endorsements = raw_key.get("endorsements")
    if endorsements and "msteams" not in endorsements:
        logger.warning(
            f"Connector key {raw_key.get('kid')} is not endorsed for msteams "
            f"(endorsements={endorsements}); allowing while endorsement "
            f"enforcement is in observation mode"
        )


def verify_service_url(claims: dict[str, Any], activity_service_url: str) -> None:
    """Bind an authenticated token to the body it arrived with.

    The Connector puts the ``serviceUrl`` it will accept replies for inside the
    token. Comparing it against the activity's own ``serviceUrl`` is what stops
    a captured token being replayed with a body pointing our replies (and the
    bearer token attached to them) at an attacker's host.

    NOTE the claim key is lowercase ``serviceurl`` on the wire even though the
    docs spell it ``serviceUrl``; reading the camelCase spelling yields
    ``None`` and silently disables this check entirely.
    """
    claimed = claims.get("serviceurl")
    if not claimed:
        raise TeamsAuthError("token carries no serviceurl claim")
    if claimed.rstrip("/") != (activity_service_url or "").rstrip("/"):
        raise TeamsAuthError("activity serviceUrl does not match the token")


def is_allowed_service_url(service_url: str) -> bool:
    """Whether we will attach our outbound bearer token to this URL.

    Defence in depth behind :func:`verify_service_url`: even a correctly
    signed token only ever directs us at Connector hosts, over HTTPS.
    """
    try:
        parsed = urlparse(service_url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    # The Agents Playground runs on the developer's own machine, so its
    # serviceUrl is loopback (plain http) — or the container's alias for it
    # once :func:`rewrite_loopback_for_container` has redirected it. Only
    # reachable in the double-gated local mode, where no token is attached to
    # the request anyway.
    if host in _LOOPBACK_HOSTS or host == _CONTAINER_HOST_ALIAS:
        return config.allow_unverified_requests()
    if parsed.scheme != "https":
        return False  # Never send a bearer token in cleartext.
    return host in config.ALLOWED_SERVICE_URL_HOSTS or any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in config.ALLOWED_SERVICE_URL_SUFFIXES
    )


def is_fetchable_attachment_url(url: str) -> bool:
    """Whether we will issue a server-side GET for this attachment.

    Attachment URLs ride in the activity body, which the Connector token does
    not sign — the same reason ``serviceUrl`` is bound to the token. A fetch
    with no allowlist is a request generator pointed at our own network, so
    require HTTPS and refuse addresses inside our own network.

    This is the cheap pre-filter only: it reads the URL, and the URL cannot
    tell us where a name points. :func:`ensure_attachment_host_is_external`
    is the actual boundary, applied at fetch time.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in _LOOPBACK_HOSTS or host == _CONTAINER_HOST_ALIAS:
        # Playground attachments are served off the developer's own machine.
        return config.allow_unverified_requests()
    if parsed.scheme != "https":
        return False
    return not _is_internal_host(host)


async def ensure_attachment_host_is_external(url: str) -> None:
    """Raise unless every address ``url``'s host resolves to is off our network.

    The text check cannot catch the interesting cases: ``127.1`` and
    ``2130706433`` are not valid literals to :mod:`ipaddress` but
    ``getaddrinfo`` resolves both to loopback, and a plain DNS name can point
    anywhere at all.

    Resolution still races the connect that follows it, so a name that flips
    between the two would slip through; closing that needs peer-address
    validation inside the transport. What this removes is every bypass that
    costs an attacker nothing.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("attachment URL has no host")
    if (host in _LOOPBACK_HOSTS or host == _CONTAINER_HOST_ALIAS) and (
        config.allow_unverified_requests()
    ):
        return  # Playground attachments are served from the developer's machine.
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, parsed.port or 443, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as exc:
        raise ValueError(f"attachment host {host!r} does not resolve") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or any(_is_internal_host(address) for address in addresses):
        raise ValueError(f"attachment host {host!r} resolves inside our network")


def _is_internal_host(host: str) -> bool:
    """True for a literal address inside a range we must never dial.

    Only literals are decided here: a name that *resolves* into one of these
    ranges still gets through, which is why the bearer-bearing path uses the
    stricter Connector allowlist instead of this check.
    """
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def rewrite_loopback_for_container(service_url: str) -> str:
    """Point a loopback ``serviceUrl`` at the container's host instead.

    The Playground advertises itself as ``localhost``, which from inside Docker
    resolves to the container, so replies never reach it. Gated on local mode
    plus being in a container, and must run *after*
    :func:`verify_service_url`, which compares the URL as it arrived.
    """
    if not config.allow_unverified_requests() or not _in_container():
        return service_url
    parsed = urlparse(service_url)
    if (parsed.hostname or "").lower() not in _LOOPBACK_HOSTS:
        return service_url
    netloc = _CONTAINER_HOST_ALIAS
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _in_container() -> bool:
    """Whether this process is running inside a Docker container."""
    return os.path.exists("/.dockerenv")


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise TeamsAuthError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise TeamsAuthError("Authorization header is not a bearer token")
    return token.strip()
