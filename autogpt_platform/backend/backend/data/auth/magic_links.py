"""
Magic link service for email verification and password reset.

Magic links are single-use, time-limited tokens sent via email that allow
users to verify their email address or reset their password without entering
the old password.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from prisma.models import UserAuthMagicLink
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Magic link TTLs
EMAIL_VERIFICATION_TTL = timedelta(hours=24)
PASSWORD_RESET_TTL = timedelta(minutes=15)

# Token prefix for identification
MAGIC_LINK_PREFIX = "agpt_ml_"


class MagicLinkPurpose(str, Enum):
    """Purpose of the magic link."""

    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"


class MagicLinkInfo(BaseModel):
    """Information about a valid magic link."""

    email: str
    purpose: MagicLinkPurpose
    user_id: Optional[str] = None  # Set for password reset, not for signup verification


def generate_magic_link_token() -> str:
    """
    Generate a cryptographically secure magic link token.

    Returns:
        A prefixed random token string.
    """
    random_bytes = secrets.token_urlsafe(32)
    return f"{MAGIC_LINK_PREFIX}{random_bytes}"


def hash_magic_link_token(token: str) -> str:
    """
    Hash a magic link token for storage.

    Uses SHA256 for deterministic lookup.

    Args:
        token: The plaintext magic link token.

    Returns:
        The SHA256 hex digest.
    """
    return hashlib.sha256(token.encode()).hexdigest()


async def create_magic_link(
    email: str,
    purpose: MagicLinkPurpose,
    user_id: Optional[str] = None,
) -> str:
    """
    Create a magic link token and store it in the database.

    Args:
        email: The email address associated with the link.
        purpose: The purpose of the magic link.
        user_id: Optional user ID (for password reset).

    Returns:
        The plaintext magic link token.
    """
    token = generate_magic_link_token()
    token_hash = hash_magic_link_token(token)

    # Determine TTL based on purpose
    if purpose == MagicLinkPurpose.PASSWORD_RESET:
        ttl = PASSWORD_RESET_TTL
    else:
        ttl = EMAIL_VERIFICATION_TTL

    expires_at = datetime.now(timezone.utc) + ttl

    # Invalidate any existing magic links for this email and purpose
    await UserAuthMagicLink.prisma().update_many(
        where={
            "email": email,
            "purpose": purpose.value,
            "usedAt": None,
        },
        data={"usedAt": datetime.now(timezone.utc)},  # Mark as used to invalidate
    )

    # Create new magic link
    await UserAuthMagicLink.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "tokenHash": token_hash,
            "email": email,
            "purpose": purpose.value,
            "userId": user_id,
            "expiresAt": expires_at,
        }
    )

    return token


async def validate_magic_link(
    token: str,
    expected_purpose: Optional[MagicLinkPurpose] = None,
) -> Optional[MagicLinkInfo]:
    """
    Validate a magic link token without consuming it.

    Args:
        token: The plaintext magic link token.
        expected_purpose: Optional expected purpose to validate against.

    Returns:
        MagicLinkInfo if valid, None otherwise.
    """
    token_hash = hash_magic_link_token(token)

    where_clause: dict = {
        "tokenHash": token_hash,
        "usedAt": None,
        "expiresAt": {"gt": datetime.now(timezone.utc)},
    }

    if expected_purpose:
        where_clause["purpose"] = expected_purpose.value

    db_link = await UserAuthMagicLink.prisma().find_first(where=where_clause)

    if not db_link:
        return None

    return MagicLinkInfo(
        email=db_link.email,
        purpose=MagicLinkPurpose(db_link.purpose),
        user_id=db_link.userId,
    )


async def consume_magic_link(
    token: str,
    expected_purpose: Optional[MagicLinkPurpose] = None,
) -> Optional[MagicLinkInfo]:
    """
    Validate and consume a magic link token (single-use).

    Args:
        token: The plaintext magic link token.
        expected_purpose: Optional expected purpose to validate against.

    Returns:
        MagicLinkInfo if valid and successfully consumed, None otherwise.
    """
    # First validate
    link_info = await validate_magic_link(token, expected_purpose)
    if not link_info:
        return None

    # Then consume (mark as used)
    token_hash = hash_magic_link_token(token)
    result = await UserAuthMagicLink.prisma().update_many(
        where={
            "tokenHash": token_hash,
            "usedAt": None,
        },
        data={"usedAt": datetime.now(timezone.utc)},
    )

    if result == 0:
        # Race condition - link was consumed by another request
        logger.warning("Magic link was already consumed (race condition)")
        return None

    return link_info


async def create_email_verification_link(email: str) -> str:
    """
    Create an email verification magic link.

    Args:
        email: The email address to verify.

    Returns:
        The plaintext magic link token.
    """
    return await create_magic_link(email, MagicLinkPurpose.EMAIL_VERIFICATION)


async def create_password_reset_link(email: str, user_id: str) -> str:
    """
    Create a password reset magic link.

    Args:
        email: The user's email address.
        user_id: The user's ID.

    Returns:
        The plaintext magic link token.
    """
    return await create_magic_link(
        email, MagicLinkPurpose.PASSWORD_RESET, user_id=user_id
    )


async def verify_email_token(token: str) -> Optional[str]:
    """
    Verify an email verification token and consume it.

    Args:
        token: The magic link token.

    Returns:
        The email address if valid, None otherwise.
    """
    link_info = await consume_magic_link(token, MagicLinkPurpose.EMAIL_VERIFICATION)
    return link_info.email if link_info else None


async def verify_password_reset_token(token: str) -> Optional[tuple[str, str]]:
    """
    Verify a password reset token and consume it.

    Args:
        token: The magic link token.

    Returns:
        Tuple of (user_id, email) if valid, None otherwise.
    """
    link_info = await consume_magic_link(token, MagicLinkPurpose.PASSWORD_RESET)
    if not link_info or not link_info.user_id:
        return None
    return link_info.user_id, link_info.email
