"""
JWT token generation and validation for user authentication.

This module generates tokens compatible with Supabase JWT format to ensure
a smooth migration without requiring frontend changes.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from prisma.models import UserAuthRefreshToken
from pydantic import BaseModel

from autogpt_libs.auth.config import get_settings

logger = logging.getLogger(__name__)

# Token TTLs
ACCESS_TOKEN_TTL = timedelta(hours=1)
REFRESH_TOKEN_TTL = timedelta(days=30)

# Refresh token prefix for identification
REFRESH_TOKEN_PREFIX = "agpt_rt_"


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    expires_in: int  # seconds until access token expires
    token_type: str = "bearer"


class JWTPayload(BaseModel):
    """JWT payload structure matching Supabase format."""

    sub: str  # user ID
    email: str
    phone: str = ""
    role: str = "authenticated"
    aud: str = "authenticated"
    iat: int  # issued at (unix timestamp)
    exp: int  # expiration (unix timestamp)


def create_access_token(
    user_id: str,
    email: str,
    role: str = "authenticated",
    phone: str = "",
) -> str:
    """
    Create a JWT access token.

    The token format matches Supabase JWT structure so existing backend
    validation code continues to work without modification.

    Args:
        user_id: The user's UUID.
        email: The user's email address.
        role: The user's role (default: "authenticated").
        phone: The user's phone number (optional).

    Returns:
        The encoded JWT token string.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "email": email,
        "phone": phone,
        "role": role,
        "aud": "authenticated",
        "iat": int(now.timestamp()),
        "exp": int((now + ACCESS_TOKEN_TTL).timestamp()),
    }

    return jwt.encode(payload, settings.JWT_VERIFY_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[JWTPayload]:
    """
    Decode and validate a JWT access token.

    Args:
        token: The JWT token string.

    Returns:
        The decoded payload if valid, None otherwise.
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.JWT_VERIFY_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience="authenticated",
        )
        return JWTPayload(**payload)
    except jwt.ExpiredSignatureError:
        logger.debug("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None


def generate_refresh_token() -> str:
    """
    Generate a cryptographically secure refresh token.

    Returns:
        A prefixed random token string.
    """
    random_bytes = secrets.token_urlsafe(32)
    return f"{REFRESH_TOKEN_PREFIX}{random_bytes}"


def hash_refresh_token(token: str) -> str:
    """
    Hash a refresh token for storage.

    Uses SHA256 for deterministic lookup (unlike passwords which use Argon2).

    Args:
        token: The plaintext refresh token.

    Returns:
        The SHA256 hex digest.
    """
    return hashlib.sha256(token.encode()).hexdigest()


async def create_refresh_token_db(
    user_id: str,
    token: Optional[str] = None,
) -> tuple[str, datetime]:
    """
    Create a refresh token and store it in the database.

    Args:
        user_id: The user's UUID.
        token: Optional pre-generated token (used in OAuth flow).

    Returns:
        Tuple of (plaintext token, expiration datetime).
    """
    if token is None:
        token = generate_refresh_token()

    token_hash = hash_refresh_token(token)
    expires_at = datetime.now(timezone.utc) + REFRESH_TOKEN_TTL

    await UserAuthRefreshToken.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "tokenHash": token_hash,
            "userId": user_id,
            "expiresAt": expires_at,
        }
    )

    return token, expires_at


async def validate_refresh_token(token: str) -> Optional[str]:
    """
    Validate a refresh token and return the associated user ID.

    Args:
        token: The plaintext refresh token.

    Returns:
        The user ID if valid, None otherwise.
    """
    token_hash = hash_refresh_token(token)

    db_token = await UserAuthRefreshToken.prisma().find_first(
        where={
            "tokenHash": token_hash,
            "revokedAt": None,
            "expiresAt": {"gt": datetime.now(timezone.utc)},
        }
    )

    if not db_token:
        return None

    return db_token.userId


async def revoke_refresh_token(token: str) -> bool:
    """
    Revoke a refresh token.

    Args:
        token: The plaintext refresh token.

    Returns:
        True if a token was revoked, False otherwise.
    """
    token_hash = hash_refresh_token(token)

    result = await UserAuthRefreshToken.prisma().update_many(
        where={
            "tokenHash": token_hash,
            "revokedAt": None,
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )

    return result > 0


async def revoke_all_user_refresh_tokens(user_id: str) -> int:
    """
    Revoke all refresh tokens for a user.

    Used for global logout or security events.

    Args:
        user_id: The user's UUID.

    Returns:
        Number of tokens revoked.
    """
    result = await UserAuthRefreshToken.prisma().update_many(
        where={
            "userId": user_id,
            "revokedAt": None,
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )

    return result


async def create_token_pair(
    user_id: str,
    email: str,
    role: str = "authenticated",
) -> TokenPair:
    """
    Create a complete token pair (access + refresh).

    Args:
        user_id: The user's UUID.
        email: The user's email.
        role: The user's role.

    Returns:
        TokenPair with access_token, refresh_token, and metadata.
    """
    access_token = create_access_token(user_id, email, role)
    refresh_token, _ = await create_refresh_token_db(user_id)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )
