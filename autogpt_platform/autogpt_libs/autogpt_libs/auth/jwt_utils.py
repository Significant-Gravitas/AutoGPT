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
        return payload
    except jwt.ExpiredSignatureError as e:
        raise ValueError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}") from e


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
