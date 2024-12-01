from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitSettings(BaseSettings):
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for rate limiting",
        validation_alias="RATE_LIMIT_REDIS_URL"
    )

    requests_per_minute: int = Field(
        default=60,
        description="Maximum number of requests allowed per minute per API key",
        validation_alias="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )

    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")


RATE_LIMIT_SETTINGS = RateLimitSettings()
