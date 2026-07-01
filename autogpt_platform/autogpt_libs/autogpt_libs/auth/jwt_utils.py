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
        payload = parse_jwt_token(credentials.credentials)
        logger.debug("Token decoded successfully")
        return payload
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


def parse_jwt_token(token: str) -> dict[str, Any]:
    """
    Parse and validate a JWT token.

    Symmetric (HS*) tokens are verified with the shared secret
    (`JWT_VERIFY_KEY`); asymmetric tokens are verified against the JWK set
    published by the platform auth service (`JWT_JWKS_URL`). Both paths can be
    active at once, which keeps sessions issued by a previous auth provider
    valid during a migration window.

    :param token: The token to parse
    :return: The decoded payload
    :raises ValueError: If the token is invalid or expired
    """
    settings = get_settings()
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}")

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
        except jwt.PyJWKClientError as e:
            raise ValueError(f"Invalid token: {str(e)}")
        algorithms = settings.JWT_JWKS_ALGORITHMS

    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}")


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
            )
            _jwks_client_url = url
    return _jwks_client


def verify_user(jwt_payload: dict | None, admin_only: bool) -> User:
    if jwt_payload is None:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    user_id = jwt_payload.get("sub")

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    if admin_only and jwt_payload["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return User.from_payload(jwt_payload)
