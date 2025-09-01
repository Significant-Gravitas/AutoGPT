import hashlib
import secrets
from typing import NamedTuple

from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class APIKeyContainer(NamedTuple):
    """Container for API key parts."""

    key: str
    head: str
    tail: str
    hash: str
    salt: str


class APIKeySmith:
    PREFIX: str = "agpt_"
    HEAD_LENGTH: int = 8
    TAIL_LENGTH: int = 8

    def generate_key(self) -> APIKeyContainer:
        """Generate a new API key with secure hashing."""
        raw_key = f"{self.PREFIX}{secrets.token_urlsafe(32)}"
        hash, salt = self.hash_key(raw_key)

        return APIKeyContainer(
            key=raw_key,
            head=raw_key[: self.HEAD_LENGTH],
            tail=raw_key[-self.TAIL_LENGTH :],
            hash=hash,
            salt=salt,
        )

    def verify_key(
        self, provided_key: str, known_hash: str, known_salt: str | None = None
    ) -> bool:
        """
        Verify an API key against a known hash (+ salt).
        Supports verifying both legacy SHA256 and secure Scrypt hashes.
        """
        if not provided_key.startswith(self.PREFIX):
            return False

        # Handle legacy SHA256 hashes (migration support)
        if known_salt is None:
            legacy_hash = hashlib.sha256(provided_key.encode()).hexdigest()
            return secrets.compare_digest(legacy_hash, known_hash)

        try:
            salt_bytes = bytes.fromhex(known_salt)
            provided_hash = self._hash_key_with_salt(provided_key, salt_bytes)
            return secrets.compare_digest(provided_hash, known_hash)
        except (ValueError, TypeError):
            return False

    def hash_key(self, raw_key: str) -> tuple[str, str]:
        """Migrate a legacy hash to secure hash format."""
        salt = self._generate_salt()
        hash = self._hash_key_with_salt(raw_key, salt)
        return hash, salt.hex()

    def _generate_salt(self) -> bytes:
        """Generate a random salt for hashing."""
        return secrets.token_bytes(32)

    def _hash_key_with_salt(self, raw_key: str, salt: bytes) -> str:
        """Hash API key using Scrypt with salt."""
        kdf = Scrypt(
            length=32,
            salt=salt,
            n=2**14,  # CPU/memory cost parameter
            r=8,  # Block size parameter
            p=1,  # Parallelization parameter
        )
        key_hash = kdf.derive(raw_key.encode())
        return key_hash.hex()
