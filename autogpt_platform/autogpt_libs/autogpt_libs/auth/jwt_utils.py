import asyncio
import logging
import threading
from typing import Any

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import get_settings
from .models import User

logger = logging.getLogger(__name__)

# Bearer token authentication scheme
bearer_jwt_auth = HTTPBearer(
    bearerFormat="jwt", scheme_name="HTTPBearerJWT", auto_error=False
)

# Refresh the cached JWK set hourly; key rotation keeps old keys in the set
# during the grace period, so a stale cache only matters for brand-new keys
# (handled below by PyJWKClient's kid-miss refetch).
JWKS_CACHE_LIFESPAN_SECONDS = 3600

# Upper bound on a single JWKS fetch. The JWKS endpoint is our own frontend, so
# a slow response means it is redeploying or unhealthy — fail the request
# quickly rather than tying up a worker for PyJWT's 30s default.
JWKS_FETCH_TIMEOUT_SECONDS = 5

# Cached client keyed on the JWKS URL: if the URL changes (config reload,
# test override), the old client is discarded instead of silently serving
# keys from the previous endpoint.
_jwks_client: jwt.PyJWKClient | None = None
_jwks_client_url: str | None = None
_jwks_client_lock = threading.Lock()


async def get_jwt_payload(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_jwt_auth),
) -> dict[str, Any]:
    """
    Extract and validate JWT payload from HTTP Authorization header.

    This is the core authentication function that handles:
    - Reading the `Authorization` header to obtain the JWT token
    - Verifying the JWT token's signature
    - Decoding the JWT token's payload

    :param credentials: HTTP Authorization credentials from bearer token
    :return: JWT payload dictionary
    :raises HTTPException: 401 if authentication fails
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    try:
        payload = await parse_jwt_token_async(credentials.credentials)
        logger.debug("Token decoded successfully")
        return payload
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


async def parse_jwt_token_async(
    token: str, audience: str = "authenticated"
) -> dict[str, Any]:
    """Async wrapper around :func:`parse_jwt_token`.

    On a JWKS cache miss the verification does a *synchronous* HTTP fetch
    (PyJWKClient uses urllib). Awaiting that directly on the event loop stalls
    every other request on the worker until it returns, not just this one — so
    hand it to a thread. Cache hits are pure CPU and return immediately.
    """
    return await asyncio.to_thread(parse_jwt_token, token, audience)


def parse_jwt_token(token: str, audience: str = "authenticated") -> dict[str, Any]:
    """
    Parse and validate a JWT token.

    Symmetric (HS*) tokens are verified with the shared secret
    (`JWT_VERIFY_KEY`); asymmetric tokens are verified against the JWK set
    published by the platform auth service (`JWT_JWKS_URL`). Both paths can be
    active at once, which keeps sessions issued by a previous auth provider
    valid during a migration window.

    :param token: The token to parse
    :param audience: The `aud` claim the token must carry. Defaults to the
        user-token audience; service tokens use a distinct audience so the
        two planes can't be replayed against each other.
    :return: The decoded payload
    :raises ValueError: If the token is invalid or expired
    """
    settings = get_settings()
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}") from e

    algorithm = header.get("alg", "")
    if algorithm.startswith("HS"):
        if not settings.JWT_VERIFY_KEY:
            raise ValueError("Invalid token: symmetric tokens are not accepted")
        key = settings.JWT_VERIFY_KEY
        algorithms = [settings.JWT_ALGORITHM]
    else:
        if not settings.JWT_JWKS_URL:
            raise ValueError("Invalid token: asymmetric tokens are not accepted")
        try:
            key = _get_jwks_client().get_signing_key_from_jwt(token).key
            algorithms = settings.JWT_JWKS_ALGORITHMS
        except jwt.PyJWKClientError as e:
            # The legacy verifier supported — and its config text recommended —
            # asymmetric algorithms, with the public key in JWT_VERIFY_KEY. A
            # token whose kid isn't in the Better Auth JWK set can therefore
            # still be a live legacy session from that configuration, so the
            # migration-window grace extends here too: fall back to the shared
            # legacy key when it's configured for a matching asymmetric alg.
            if (
                settings.JWT_VERIFY_KEY
                and not settings.JWT_ALGORITHM.startswith("HS")
                and algorithm == settings.JWT_ALGORITHM
            ):
                key = settings.JWT_VERIFY_KEY
                algorithms = [settings.JWT_ALGORITHM]
            else:
                raise ValueError(f"Invalid token: {str(e)}") from e

    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=audience,
        )
        # REL-001 revocation: if jti/sid is in Redis denylist, reject.
        # Fail-open on Redis outage (bounded 5m exposure, not outage).
        try:
            if _is_jti_revoked(payload):
                raise ValueError("Token has been revoked")
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Revocation check failed (fail-open): {e}")
        return payload
    except jwt.ExpiredSignatureError as e:
        raise ValueError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}") from e


# REL-001 revocation TTL: 5m (300s) matches auth.ts expirationTime "5m"
# and cookieCache maxAge 5*60. The denylist only needs to cover already-
# issued JWTs plus the cookieCache window. Session rows are deleted on
# logout/revokeSessionsOnPasswordReset so no *new* JWTs can be minted
# after revocation; 300s covers the last JWT's remaining validity.
# Deployments that fear clock skew or want to cover multiple rotation
# cycles may raise sid TTL to 86400 (24h) at the cost of one extra
# Redis key per logout (still cheap). jti always stays 300.
REVOKED_JTI_TTL_SECONDS = 300
REVOKED_SID_TTL_SECONDS = 300  # alternative: 86400 if skew-conscious


def _get_redis_client():  # type: ignore[no-untyped-def]
    """Resolve the shared Redis client. Tries backend clusters then util cache.

    Prefers ``backend.data.redis_client.get_redis`` (canonical cluster client
    with host/port from env, decode_responses=True). Falls back to
    ``backend.util.cache._get_redis`` for test harnesses that only configure
    the util cache. Raises on failure so callers can fail-open.
    """
    last_err: Exception | None = None
    for importer in (
        lambda: __import__(
            "backend.data.redis_client", fromlist=["get_redis"]
        ).get_redis(),  # type: ignore[no-redef]
        lambda: __import__(
            "backend.util.cache", fromlist=["_get_redis"]
        )._get_redis(),  # type: ignore[no-redef]
    ):
        try:
            return importer()
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    # Final fallback: legacy path some configs still reference
    try:
        from backend.data.redis import get_redis  # type: ignore

        return get_redis()
    except Exception as e:  # noqa: BLE001
        if last_err is not None:
            raise last_err from e
        raise


def revoke_jti(jti: str, ttl_seconds: int = REVOKED_JTI_TTL_SECONDS) -> bool:
    """Write a jti denylist marker: revoked:jti:{jti} EX ttl.

    Returns True if the marker was written, False on Redis failure
    (fail-open caller should log). ttl 300 matches JWT 5m expiry.
    """
    if not jti:
        return False
    try:
        r = _get_redis_client()
        # RedisCluster with decode_responses=True returns str; binary mode
        # returns bytes — both are truthy checks in _is_jti_revoked, so either
        # write form is fine. Use string value.
        r.setex(f"revoked:jti:{jti}", ttl_seconds, "1")
        logger.info("Revoked jti %s for %ss", jti, ttl_seconds)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("revoke_jti failed (fail-open, bounded 5m): %s", e)
        return False


def revoke_sid(sid: str, ttl_seconds: int = REVOKED_SID_TTL_SECONDS) -> bool:
    """Write a sid denylist marker: revoked:sid:{sid} EX ttl.

    Revoking the sid invalidates *all* JWTs minted from that session
    for the remainder of their 5m validity. TTL 300 is sufficient
    because the session row is already deleted — no new JWTs will be
    minted for a dead session, so the denylist only guards the last
    cohort. Use 86400 if you want to guard against a stolen
    session_data cookie replay after JWT expiry (defense in depth).
    """
    if not sid:
        return False
    try:
        r = _get_redis_client()
        r.setex(f"revoked:sid:{sid}", ttl_seconds, "1")
        logger.info("Revoked sid %s for %ss", sid, ttl_seconds)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("revoke_sid failed (fail-open, bounded 5m): %s", e)
        return False


def revoke_token_payload(
    payload: dict[str, Any],
    jti_ttl: int = REVOKED_JTI_TTL_SECONDS,
    sid_ttl: int = REVOKED_SID_TTL_SECONDS,
) -> bool:
    """Revoke both jti and sid from a decoded payload. Writes pipeline if both.

    Returns True if at least one marker was written.
    """
    jti = payload.get("jti")
    sid = payload.get("sid")
    if not jti and not sid:
        return False
    did = False
    # Pipeline when both present to save a round-trip
    if jti and sid:
        try:
            r = _get_redis_client()
            pipe = r.pipeline() if hasattr(r, "pipeline") else None
            if pipe is not None:
                pipe.setex(f"revoked:jti:{jti}", jti_ttl, "1")
                pipe.setex(f"revoked:sid:{sid}", sid_ttl, "1")
                pipe.execute()
                logger.info(
                    "Revoked jti %s + sid %s (jti %ss, sid %ss)",
                    jti,
                    sid,
                    jti_ttl,
                    sid_ttl,
                )
                return True
        except Exception as e:  # noqa: BLE001
            logger.warning("revoke_token_payload pipeline failed: %s", e)
            # fall through to individual writes
    if jti:
        did = revoke_jti(jti, jti_ttl) or did
    if sid:
        did = revoke_sid(sid, sid_ttl) or did
    return did


def _is_jti_revoked(payload: dict[str, Any]) -> bool:
    """Check Redis denylist for jti/sid. Returns True if revoked.

    Fail-open on Redis outage: bounded 5m exposure (JWT expiry) rather
    than a full outage. Callers in parse_jwt_token log the exception
    at warning level.
    """
    jti = payload.get("jti")
    sid = payload.get("sid")
    if not jti and not sid:
        return False
    # Lazy import to avoid hard dependency at import time
    try:
        r = _get_redis_client()
        if jti:
            val = r.get(f"revoked:jti:{jti}")
            # Redis may return bytes (binary mode), str, or int
            if val is not None and val is not False and val != 0:
                # Handle bytes "1" / b"1" vs integer 1
                if isinstance(val, bytes):
                    if val not in (b"0", b""):
                        return True
                elif val not in ("0", "", 0):
                    return True
        if sid:
            val = r.get(f"revoked:sid:{sid}")
            if val is not None and val is not False and val != 0:
                if isinstance(val, bytes):
                    if val not in (b"0", b""):
                        return True
                elif val not in ("0", "", 0):
                    return True
        return False
    except Exception:
        # No redis or not connected — fail-open (bounded by 5m expiry)
        return False


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client, _jwks_client_url

    url = get_settings().JWT_JWKS_URL
    if _jwks_client is not None and _jwks_client_url == url:
        return _jwks_client

    with _jwks_client_lock:
        if _jwks_client is None or _jwks_client_url != url:
            _jwks_client = jwt.PyJWKClient(
                url,
                cache_keys=True,
                lifespan=JWKS_CACHE_LIFESPAN_SECONDS,
                # PyJWT defaults to 30s. The fetch is synchronous, so on a
                # cache miss that is 30s of a worker doing nothing — bound it
                # to something closer to "the frontend is briefly redeploying".
                timeout=JWKS_FETCH_TIMEOUT_SECONDS,
            )
            _jwks_client_url = url
    return _jwks_client


def verify_user(jwt_payload: dict | None, admin_only: bool) -> User:
    if jwt_payload is None:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    user_id = jwt_payload.get("sub")

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    if admin_only and jwt_payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return User.from_payload(jwt_payload)
