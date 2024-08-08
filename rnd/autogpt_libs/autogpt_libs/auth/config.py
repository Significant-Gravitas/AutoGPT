import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    JWT_SECRET_KEY: str = os.getenv("SUPABASE_JWT_SECRET", "")
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    JWT_ALGORITHM: str = "HS256"

    @property
    def is_configured(self) -> bool:
        return bool(self.JWT_SECRET_KEY)


settings = Settings()
