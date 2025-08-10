from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    launch_darkly_sdk_key: str = Field(
        default="",
        description="The Launch Darkly SDK key",
        validation_alias="LAUNCH_DARKLY_SDK_KEY",
    )

    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")


SETTINGS = Settings()
