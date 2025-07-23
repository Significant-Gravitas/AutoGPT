import os


class Settings:
    def __init__(self):
        self.JWT_SECRET_KEY: str = os.getenv("SUPABASE_JWT_SECRET", "")
        self.ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
        self.JWT_ALGORITHM: str = "HS256"

    @property
    def is_configured(self) -> bool:
        return bool(self.JWT_SECRET_KEY)


settings = Settings()
