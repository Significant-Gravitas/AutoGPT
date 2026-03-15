"""Password hashing utilities for the ABN Co-Navigator user auth."""
from __future__ import annotations

import hashlib
import os


def hash_password(password: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hash string: '<salt_hex>:<dk_hex>'."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + ":" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    """Return True if *password* matches the stored hash."""
    try:
        salt_hex, dk_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        return dk.hex() == dk_hex
    except Exception:
        return False
