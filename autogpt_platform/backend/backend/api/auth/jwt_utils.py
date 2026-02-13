import logging
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
        payload = jwt.decode(
            token,
            settings.JWT_VERIFY_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience="authenticated",
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
