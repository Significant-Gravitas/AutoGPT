import logging
import os

logger = logging.getLogger(__name__)


class AuthConfigError(ValueError):
    """Raised when authentication configuration is invalid."""

    pass


class Settings:
    def __init__(self):
        self.JWT_SECRET_KEY: str = os.getenv("SUPABASE_JWT_SECRET", "")
        self.JWT_ALGORITHM: str = "HS256"

        if not self.JWT_SECRET_KEY:
            raise AuthConfigError(
                "SUPABASE_JWT_SECRET must be set. "
                "An empty JWT secret would allow anyone to forge valid tokens."
            )

        if len(self.JWT_SECRET_KEY) < 32:
            logger.warning(
                "⚠️ SUPABASE_JWT_SECRET appears weak (less than 32 characters). "
                "Consider using a longer, cryptographically secure secret."
            )


settings = Settings()
