import json
import os
from typing import Any, Dict, Generic, Set, Tuple, Type, TypeVar
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from autogpt_server.util.data import get_config_path, get_data_path, get_secrets_path

T = TypeVar("T", bound=BaseSettings)


class UpdateTrackingModel(BaseModel, Generic[T]):
    _updated_fields: Set[str] = PrivateAttr(default_factory=set)

    def __setattr__(self, name: str, value) -> None:
        if name in self.model_fields:
            self._updated_fields.add(name)
        super().__setattr__(name, value)

    def mark_updated(self, field_name: str) -> None:
        if field_name in self.model_fields:
            self._updated_fields.add(field_name)

    def clear_updates(self) -> None:
        self._updated_fields.clear()

    def get_updates(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self._updated_fields}

    @property
    def updated_fields(self):
        return self._updated_fields


class Config(UpdateTrackingModel["Config"], BaseSettings):
    """Config for the server."""

    num_workers: int = Field(
        default=9, ge=1, le=100, description="Number of workers to use for execution."
    )
    # Add more configuration fields as needed

    model_config = SettingsConfigDict(
        json_file=[
            get_config_path() / "config.default.json",
            get_config_path() / "config.json",
        ],
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (JsonConfigSettingsSource(settings_cls),)


class Secrets(UpdateTrackingModel["Secrets"], BaseSettings):
    """Secrets for the server."""
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    
    reddit_client_id: str = Field(default="", description="Reddit client ID")
    reddit_client_secret: str = Field(default="", description="Reddit client secret")
    reddit_username: str = Field(default="", description="Reddit username")
    reddit_password: str = Field(default="", description="Reddit password")
    
    # Add more secret fields as needed

    model_config = SettingsConfigDict(
        secrets_dir=get_secrets_path(),
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


class Settings(BaseModel):
    config: Config = Config()
    secrets: Secrets = Secrets()

    def save(self) -> None:
        # Save updated config to JSON file
        if self.config.updated_fields:
            config_to_save = self.config.get_updates()
            config_path = os.path.join(get_data_path(), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r+") as f:
                    existing_config: Dict[str, Any] = json.load(f)
                    existing_config.update(config_to_save)
                    f.seek(0)
                    json.dump(existing_config, f, indent=2)
                    f.truncate()
            else:
                with open(config_path, "w") as f:
                    json.dump(config_to_save, f, indent=2)
            self.config.clear_updates()

        # Save updated secrets to individual files
        secrets_dir = get_secrets_path()
        for key in self.secrets.updated_fields:
            secret_file = os.path.join(secrets_dir, key)
            with open(secret_file, "w") as f:
                f.write(str(getattr(self.secrets, key)))
        self.secrets.clear_updates()
