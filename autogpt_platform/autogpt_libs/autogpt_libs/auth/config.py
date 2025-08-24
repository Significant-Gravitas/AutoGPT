import logging
import os

logger = logging.getLogger(__name__)


class AuthConfigError(Exception):
    """Raised when authentication configuration is invalid."""

    pass


class Settings:
    def __init__(self):
        self.JWT_SECRET_KEY: str = os.getenv("SUPABASE_JWT_SECRET", "")
        self.ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
        self.JWT_ALGORITHM: str = "HS256"

        # Critical security check: prevent JWT forgery via empty secret
        if self.ENABLE_AUTH and not self.JWT_SECRET_KEY:
            raise AuthConfigError(
                "SUPABASE_JWT_SECRET must be set when ENABLE_AUTH is true. "
                "An empty JWT secret allows anyone to forge valid tokens."
            )

        # Warn if JWT secret looks suspiciously short (less than 32 chars)
        if self.ENABLE_AUTH and len(self.JWT_SECRET_KEY) < 32:
            logger.warning(
                "⚠️ JWT secret appears weak (less than 32 characters). "
                "Consider using a longer, cryptographically secure secret."
            )

        if not self.ENABLE_AUTH:
            logger.warning("⚠️ autogpt_libs.auth disabled")


settings = Settings()
