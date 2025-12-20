import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
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


def create_access_token(
    user_id: str,
    email: str,
    role: str = "authenticated",
    email_verified: bool = False,
) -> str:
    """
    Generate a new JWT access token.

    :param user_id: The user's unique identifier
    :param email: The user's email address
    :param role: The user's role (default: "authenticated")
    :param email_verified: Whether the user's email is verified
    :return: Encoded JWT token
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "email_verified": email_verified,
        "aud": settings.JWT_AUDIENCE,
        "iss": settings.JWT_ISSUER,
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": str(uuid.uuid4()),  # Unique token ID
    }

    return jwt.encode(payload, settings.JWT_SIGN_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token() -> tuple[str, str]:
    """
    Generate a new refresh token.

    Returns a tuple of (raw_token, hashed_token).
    The raw token should be sent to the client.
    The hashed token should be stored in the database.
    """
    raw_token = secrets.token_urlsafe(64)
    hashed_token = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, hashed_token


def hash_token(token: str) -> str:
    """Hash a token using SHA-256."""
    return hashlib.sha256(token.encode()).hexdigest()


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

    :param token: The token to parse
    :return: The decoded payload
    :raises ValueError: If the token is invalid or expired
    """
    settings = get_settings()
    try:
        # Build decode options
        options = {
            "verify_aud": True,
            "verify_iss": bool(settings.JWT_ISSUER),
        }

        payload = jwt.decode(
            token,
            settings.JWT_VERIFY_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER if settings.JWT_ISSUER else None,
            options=options,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}")


def verify_user(jwt_payload: dict | None, admin_only: bool) -> User:
    if jwt_payload is None:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    user_id = jwt_payload.get("sub")

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    if admin_only and jwt_payload["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return User.from_payload(jwt_payload)
