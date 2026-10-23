"""Verify OIDC tokens that workloads exchange for short-lived AutoGPT tokens.

Signing keys come from the issuer's discovery document, fetched through the
SSRF-safe HTTP client and cached per issuer. Only the issuer a workload
identity was configured with is ever contacted; the issuer a token claims is
compared against it, never fetched.
"""

import logging
import time
from typing import Any

import aiohttp
import jwt

from backend.util.cache import cached
from backend.util.request import Requests

logger = logging.getLogger(__name__)

# Asymmetric only: a shared-secret (HS*) or unsigned token proves nothing here.
ALLOWED_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
        "EdDSA",
    }
)
CLOCK_LEEWAY_SECONDS = 60
# A token that stays valid for longer than this is a static credential with
# extra steps, and is refused.
MAX_TOKEN_LIFETIME_SECONDS = 24 * 3600

KEYS_CACHE_SECONDS = 600
# An unknown key ID can mean the issuer rotated its keys. Refetching on every
# such token would let anyone make us hammer an issuer, so refetch this rarely.
FORCED_REFRESH_SECONDS = 60
FAILED_FETCH_BACKOFF_SECONDS = 30
FETCH_TIMEOUT_SECONDS = 10
MAX_DOCUMENT_BYTES = 256 * 1024

_forced_refresh_at: dict[str, float] = {}
_failed_fetch_at: dict[str, float] = {}


class OIDCVerificationError(Exception):
    """The token is not a valid, current token from the expected issuer."""


async def verify_oidc_token(
    token: str, *, issuer: str, audience: str
) -> dict[str, Any]:
    """Return the token's claims once its signature, issuer, audience and
    lifetime check out. Raises OIDCVerificationError otherwise."""
    header = _unverified_header(token)
    key = await _signing_key(issuer, header["alg"], header.get("kid"))
    try:
        claims = jwt.decode(
            token,
            key=key.key,
            algorithms=[header["alg"]],
            audience=audience,
            issuer=issuer,
            leeway=CLOCK_LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise OIDCVerificationError(f"Token rejected: {e}") from e
    if claims["exp"] - claims["iat"] > MAX_TOKEN_LIFETIME_SECONDS:
        raise OIDCVerificationError("Token is valid for longer than 24 hours")
    return claims


def _unverified_header(token: str) -> dict[str, Any]:
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise OIDCVerificationError("Token is not a JWT") from e
    if header.get("alg") not in ALLOWED_ALGORITHMS:
        raise OIDCVerificationError(
            f"Signing algorithm {header.get('alg')!r} is not accepted"
        )
    return header


async def _signing_key(issuer: str, alg: str, kid: str | None) -> jwt.PyJWK:
    key = _select_key(await _issuer_keys(issuer), alg, kid)
    if key is None and _may_force_refresh(issuer):
        _fetch_issuer_keys.cache_delete(issuer)
        key = _select_key(await _issuer_keys(issuer), alg, kid)
    if key is None:
        raise OIDCVerificationError(
            "Token was not signed by a current key of its issuer"
        )
    return key


def _select_key(
    keys: list[dict[str, Any]], alg: str, kid: str | None
) -> jwt.PyJWK | None:
    candidates = [
        k
        for k in keys
        if k.get("kty") != "oct"
        and k.get("use", "sig") == "sig"
        and k.get("alg", alg) == alg
        and (kid is None or k.get("kid") == kid)
    ]
    # Without a key ID only an unambiguous key set identifies the signer.
    if len(candidates) != 1 and kid is None:
        return None
    for candidate in candidates:
        try:
            return jwt.PyJWK(candidate, algorithm=alg)
        except jwt.PyJWTError:
            continue
    return None


async def _issuer_keys(issuer: str) -> list[dict[str, Any]]:
    failed_at = _failed_fetch_at.get(issuer)
    if failed_at and time.monotonic() - failed_at < FAILED_FETCH_BACKOFF_SECONDS:
        raise OIDCVerificationError("Issuer's signing keys are unavailable")
    keys = await _fetch_issuer_keys(issuer)
    if keys is None:
        _remember(_failed_fetch_at, issuer)
        raise OIDCVerificationError("Could not fetch the issuer's signing keys")
    return keys


@cached(ttl_seconds=KEYS_CACHE_SECONDS, maxsize=512, cache_none=False)
async def _fetch_issuer_keys(issuer: str) -> list[dict[str, Any]] | None:
    """The issuer's published signing keys, or None if they can't be fetched."""
    try:
        if not issuer.startswith("https://"):
            raise ValueError("issuer is not an https URL")
        discovery_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
        discovery = await _fetch_json(discovery_url)
        if discovery.get("issuer") != issuer:
            raise ValueError("discovery document names a different issuer")
        jwks_uri = discovery.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri.startswith("https://"):
            raise ValueError("discovery document has no https jwks_uri")
        keys = (await _fetch_json(jwks_uri)).get("keys")
        if not isinstance(keys, list):
            raise ValueError("JWKS document has no key list")
        return [k for k in keys if isinstance(k, dict)]
    except Exception as e:
        logger.warning(f"Could not fetch signing keys of OIDC issuer {issuer}: {e}")
        return None


async def _fetch_json(url: str) -> dict[str, Any]:
    response = await Requests(retry_max_attempts=1).get(
        url,
        max_redirects=3,
        timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS),
    )
    if len(response.content) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"{url} returned more than {MAX_DOCUMENT_BYTES} bytes")
    document = response.json()
    if not isinstance(document, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return document


def _may_force_refresh(issuer: str) -> bool:
    last = _forced_refresh_at.get(issuer)
    if last and time.monotonic() - last < FORCED_REFRESH_SECONDS:
        return False
    _remember(_forced_refresh_at, issuer)
    return True


def _remember(timestamps: dict[str, float], issuer: str) -> None:
    # Issuers are user-configured, so keep these bookkeeping maps bounded.
    if len(timestamps) > 4096:
        timestamps.clear()
    timestamps[issuer] = time.monotonic()
