"""
Password hashing service using Argon2id.

OWASP 2024 recommended configuration:
- time_cost: 2 iterations
- memory_cost: 19456 KiB (19 MiB)
- parallelism: 1
"""

import logging

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from argon2.profiles import RFC_9106_LOW_MEMORY

logger = logging.getLogger(__name__)

# Use RFC 9106 low-memory profile (OWASP recommended)
# time_cost=2, memory_cost=19456, parallelism=1
_hasher = PasswordHasher.from_parameters(RFC_9106_LOW_MEMORY)


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id.

    Args:
        password: The plaintext password to hash.

    Returns:
        The hashed password string (includes algorithm params and salt).
    """
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """
    Verify a password against a hash.

    Args:
        password_hash: The stored password hash.
        password: The plaintext password to verify.

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        _hasher.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False
    except InvalidHashError:
        logger.warning("Invalid password hash format encountered")
        return False


def needs_rehash(password_hash: str) -> bool:
    """
    Check if a password hash needs to be rehashed.

    This returns True if the hash was created with different parameters
    than the current configuration, allowing for transparent upgrades.

    Args:
        password_hash: The stored password hash.

    Returns:
        True if the hash should be rehashed, False otherwise.
    """
    return _hasher.check_needs_rehash(password_hash)
