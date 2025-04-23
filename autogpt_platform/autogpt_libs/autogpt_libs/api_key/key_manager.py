import hashlib
import secrets
from typing import NamedTuple


class APIKeyContainer(NamedTuple):
    """Container for API key parts."""

    raw: str
    prefix: str
    postfix: str
    hash: str


class APIKeyManager:
    PREFIX: str = "agpt_"
    PREFIX_LENGTH: int = 8
    POSTFIX_LENGTH: int = 8

    def generate_api_key(self) -> APIKeyContainer:
        """Generate a new API key with all its parts."""
        raw_key = f"{self.PREFIX}{secrets.token_urlsafe(32)}"
        return APIKeyContainer(
            raw=raw_key,
            prefix=raw_key[: self.PREFIX_LENGTH],
            postfix=raw_key[-self.POSTFIX_LENGTH :],
            hash=hashlib.sha256(raw_key.encode()).hexdigest(),
        )

    def verify_api_key(self, provided_key: str, stored_hash: str) -> bool:
        """Verify if a provided API key matches the stored hash."""
        if not provided_key.startswith(self.PREFIX):
            return False
        return hashlib.sha256(provided_key.encode()).hexdigest() == stored_hash
