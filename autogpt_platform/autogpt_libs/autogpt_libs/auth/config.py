import logging
import os

logger = logging.getLogger(__name__)


class Settings:
    def __init__(self):
        self.JWT_SECRET_KEY: str = os.getenv("SUPABASE_JWT_SECRET", "")
        self.ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
        self.JWT_ALGORITHM: str = "HS256"

        if not self.ENABLE_AUTH:
            logger.warning("⚠️ autogpt_libs.auth disabled")


settings = Settings()
