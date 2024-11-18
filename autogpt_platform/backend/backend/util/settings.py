import json
import os
from enum import Enum
from typing import Any, Dict, Generic, List, Set, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from backend.util.data import get_data_path

T = TypeVar("T", bound=BaseSettings)


class AppEnvironment(str, Enum):
    LOCAL = "local"
    DEVELOPMENT = "dev"
    PRODUCTION = "prod"


class BehaveAs(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


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

    num_graph_workers: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of workers to use for graph execution.",
    )
    num_node_workers: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Maximum number of workers to use for node execution within a single graph.",
    )
    pyro_host: str = Field(
        default="localhost",
        description="The default hostname of the Pyro server.",
    )
    pyro_client_comm_timeout: float = Field(
        default=15,
        description="The default timeout in seconds, for Pyro client connections.",
    )
    pyro_client_comm_retry: int = Field(
        default=3,
        description="The default number of retries for Pyro client connections.",
    )
    enable_auth: bool = Field(
        default=True,
        description="If authentication is enabled or not",
    )
    enable_credit: str = Field(
        default="false",
        description="If user credit system is enabled or not",
    )
    num_user_credits_refill: int = Field(
        default=1500,
        description="Number of credits to refill for each user",
    )
    # Add more configuration fields as needed

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )

    websocket_server_host: str = Field(
        default="0.0.0.0",
        description="The host for the websocket server to run on",
    )

    websocket_server_port: int = Field(
        default=8001,
        description="The port for the websocket server to run on",
    )

    execution_manager_port: int = Field(
        default=8002,
        description="The port for execution manager daemon to run on",
    )

    execution_scheduler_port: int = Field(
        default=8003,
        description="The port for execution scheduler daemon to run on",
    )

    agent_server_port: int = Field(
        default=8004,
        description="The port for agent server daemon to run on",
    )

    database_api_port: int = Field(
        default=8005,
        description="The port for database server API to run on",
    )

    agent_api_host: str = Field(
        default="0.0.0.0",
        description="The host for agent server API to run on",
    )

    agent_api_port: int = Field(
        default=8006,
        description="The port for agent server API to run on",
    )

    frontend_base_url: str = Field(
        default="http://localhost:3000",
        description="Can be used to explicitly set the base URL for the frontend. "
        "This value is then used to generate redirect URLs for OAuth flows.",
    )

    app_env: AppEnvironment = Field(
        default=AppEnvironment.LOCAL,
        description="The name of the app environment: local or dev or prod",
    )

    behave_as: BehaveAs = Field(
        default=BehaveAs.LOCAL,
        description="What environment to behave as: local or cloud",
    )

    execution_event_bus_name: str = Field(
        default="execution_event",
        description="Name of the event bus",
    )

    trust_endpoints_for_requests: List[str] = Field(
        default_factory=list,
        description="A whitelist of trusted internal endpoints for the backend to make requests to.",
    )

    backend_cors_allow_origins: List[str] = Field(default_factory=list)

    @field_validator("backend_cors_allow_origins")
    @classmethod
    def validate_cors_allow_origins(cls, v: List[str]) -> List[str]:
        out = []
        port = None
        has_localhost = False
        has_127_0_0_1 = False
        for url in v:
            url = url.strip()
            if url.startswith(("http://", "https://")):
                if "localhost" in url:
                    port = url.split(":")[2]
                    has_localhost = True
                if "127.0.0.1" in url:
                    port = url.split(":")[2]
                    has_127_0_0_1 = True
                out.append(url)
            else:
                raise ValueError(f"Invalid URL: {url}")

        if has_127_0_0_1 and not has_localhost:
            out.append(f"http://localhost:{port}")
        if has_localhost and not has_127_0_0_1:
            out.append(f"http://127.0.0.1:{port}")

        return out

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            file_secret_settings,
            dotenv_settings,
            JsonConfigSettingsSource(settings_cls),
            init_settings,
        )


class Secrets(UpdateTrackingModel["Secrets"], BaseSettings):
    """Secrets for the server."""

    supabase_url: str = Field(default="", description="Supabase URL")
    supabase_service_role_key: str = Field(
        default="", description="Supabase service role key"
    )

    encryption_key: str = Field(default="", description="Encryption key")

    # OAuth server credentials for integrations
    # --8<-- [start:OAuthServerCredentialsExample]
    github_client_id: str = Field(default="", description="GitHub OAuth client ID")
    github_client_secret: str = Field(
        default="", description="GitHub OAuth client secret"
    )
    # --8<-- [end:OAuthServerCredentialsExample]
    google_client_id: str = Field(default="", description="Google OAuth client ID")
    google_client_secret: str = Field(
        default="", description="Google OAuth client secret"
    )
    notion_client_id: str = Field(default="", description="Notion OAuth client ID")
    notion_client_secret: str = Field(
        default="", description="Notion OAuth client secret"
    )

    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    open_router_api_key: str = Field(default="", description="Open Router API Key")

    reddit_client_id: str = Field(default="", description="Reddit client ID")
    reddit_client_secret: str = Field(default="", description="Reddit client secret")
    reddit_username: str = Field(default="", description="Reddit username")
    reddit_password: str = Field(default="", description="Reddit password")

    openweathermap_api_key: str = Field(
        default="", description="OpenWeatherMap API key"
    )

    medium_api_key: str = Field(default="", description="Medium API key")
    medium_author_id: str = Field(default="", description="Medium author ID")
    did_api_key: str = Field(default="", description="D-ID API Key")
    revid_api_key: str = Field(default="", description="revid.ai API key")
    discord_bot_token: str = Field(default="", description="Discord bot token")

    smtp_server: str = Field(default="", description="SMTP server IP")
    smtp_port: str = Field(default="", description="SMTP server port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")

    sentry_dsn: str = Field(default="", description="Sentry DSN")

    google_maps_api_key: str = Field(default="", description="Google Maps API Key")

    replicate_api_key: str = Field(default="", description="Replicate API Key")
    unreal_speech_api_key: str = Field(default="", description="Unreal Speech API Key")
    ideogram_api_key: str = Field(default="", description="Ideogram API Key")
    jina_api_key: str = Field(default="", description="Jina API Key")
    unreal_speech_api_key: str = Field(default="", description="Unreal Speech API Key")

    # Add more secret fields as needed

    model_config = SettingsConfigDict(
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
