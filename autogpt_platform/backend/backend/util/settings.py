import json
import os
from enum import Enum
from typing import Any, Dict, Generic, List, Set, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator
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
        if name in UpdateTrackingModel.model_fields:
            self._updated_fields.add(name)
        super().__setattr__(name, value)

    def mark_updated(self, field_name: str) -> None:
        if field_name in UpdateTrackingModel.model_fields:
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

    # FastAPI Thread Pool Configuration
    # IMPORTANT: FastAPI automatically offloads ALL sync functions to a thread pool:
    # - Sync endpoint functions (def instead of async def)
    # - Sync dependency functions (def instead of async def)
    # - Manually called run_in_threadpool() operations
    # Default thread pool size is only 40, which becomes a bottleneck under high concurrency
    fastapi_thread_pool_size: int = Field(
        default=60,
        ge=40,
        le=500,
        description="Thread pool size for FastAPI sync operations. All sync endpoints and dependencies automatically use this pool. Higher values support more concurrent sync operations but use more memory.",
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
        default=100,
        description="The default number of retries for Pyro client connections.",
    )
    pyro_client_max_wait: float = Field(
        default=30.0,
        description="The maximum wait time in seconds for Pyro client retries.",
    )
    rpc_client_call_timeout: int = Field(
        default=300,
        description="The default timeout in seconds, for RPC client calls.",
    )
    enable_auth: bool = Field(
        default=True,
        description="If authentication is enabled or not",
    )
    enable_credit: bool = Field(
        default=False,
        description="If user credit system is enabled or not",
    )
    enable_beta_monthly_credit: bool = Field(
        default=True,
        description="If beta monthly credits accounting is enabled or not",
    )
    num_user_credits_refill: int = Field(
        default=1500,
        description="Number of credits to refill for each user",
    )
    refund_credit_tolerance_threshold: int = Field(
        default=500,
        description="Maximum number of credits above the balance to be auto-approved.",
    )
    low_balance_threshold: int = Field(
        default=500,
        description="Credit threshold for low balance notifications (100 = $1, default 500 = $5)",
    )
    refund_notification_email: str = Field(
        default="refund@agpt.co",
        description="Email address to send refund notifications to.",
    )
    refund_request_time_key_format: str = Field(
        default="%Y-%W",  # This will allow for weekly refunds per user.
        description="Time key format for refund requests.",
    )
    execution_cost_count_threshold: int = Field(
        default=100,
        description="Number of executions after which the cost is calculated.",
    )
    execution_cost_per_threshold: int = Field(
        default=1,
        description="Cost per execution in cents after each threshold.",
    )
    execution_counter_expiration_time: int = Field(
        default=60 * 60 * 24,
        description="Time in seconds after which the execution counter is reset.",
    )
    execution_late_notification_threshold_secs: int = Field(
        default=5 * 60,
        description="Time in seconds after which the execution stuck on QUEUED status is considered late.",
    )
    cluster_lock_timeout: int = Field(
        default=300,
        description="Cluster lock timeout in seconds for graph execution coordination.",
    )
    execution_late_notification_checkrange_secs: int = Field(
        default=60 * 60,
        description="Time in seconds for how far back to check for the late executions.",
    )

    block_error_rate_threshold: float = Field(
        default=0.5,
        description="Error rate threshold (0.0-1.0) for triggering block error alerts.",
    )
    block_error_rate_check_interval_secs: int = Field(
        default=24 * 60 * 60,  # 24 hours
        description="Interval in seconds between block error rate checks.",
    )
    block_error_include_top_blocks: int = Field(
        default=3,
        description="Number of top blocks with most errors to show when no blocks exceed threshold (0 to disable).",
    )

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

    notification_service_port: int = Field(
        default=8007,
        description="The port for notification service daemon to run on",
    )

    otto_api_url: str = Field(
        default="",
        description="The URL for the Otto API service",
    )

    platform_base_url: str = Field(
        default="",
        description="Must be set so the application knows where it's hosted at. "
        "This is necessary to make sure webhooks find their way.",
    )

    frontend_base_url: str = Field(
        default="",
        description="Can be used to explicitly set the base URL for the frontend. "
        "This value is then used to generate redirect URLs for OAuth flows.",
    )

    media_gcs_bucket_name: str = Field(
        default="",
        description="The name of the Google Cloud Storage bucket for media files",
    )

    reddit_user_agent: str = Field(
        default="AutoGPT:1.0 (by /u/autogpt)",
        description="The user agent for the Reddit API",
    )

    scheduler_db_pool_size: int = Field(
        default=3,
        description="The pool size for the scheduler database connection pool",
    )

    rabbitmq_host: str = Field(
        default="localhost",
        description="The host for the RabbitMQ server",
    )
    rabbitmq_port: int = Field(
        default=5672,
        description="The port for the RabbitMQ server",
    )

    rabbitmq_vhost: str = Field(
        default="/",
        description="The vhost for the RabbitMQ server",
    )

    postmark_sender_email: str = Field(
        default="invalid@invalid.com",
        description="The email address to use for sending emails",
    )

    use_agent_image_generation_v2: bool = Field(
        default=True,
        description="Whether to use the new agent image generation service",
    )
    enable_agent_input_subtype_blocks: bool = Field(
        default=True,
        description="Whether to enable the agent input subtype blocks",
    )
    platform_alert_discord_channel: str = Field(
        default="local-alerts",
        description="The Discord channel for the platform",
    )
    product_alert_discord_channel: str = Field(
        default="product-alerts",
        description="The Discord channel for product alerts (low balance, zero balance, etc.)",
    )

    clamav_service_host: str = Field(
        default="localhost",
        description="The host for the ClamAV daemon",
    )
    clamav_service_port: int = Field(
        default=3310,
        description="The port for the ClamAV daemon",
    )
    clamav_service_timeout: int = Field(
        default=60,
        description="The timeout in seconds for the ClamAV daemon",
    )
    clamav_service_enabled: bool = Field(
        default=True,
        description="Whether virus scanning is enabled or not",
    )
    clamav_max_concurrency: int = Field(
        default=10,
        description="The maximum number of concurrent scans to perform",
    )
    clamav_mark_failed_scans_as_clean: bool = Field(
        default=False,
        description="Whether to mark failed scans as clean or not",
    )

    enable_example_blocks: bool = Field(
        default=False,
        description="Whether to enable example blocks in production",
    )

    cloud_storage_cleanup_interval_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Hours between cloud storage cleanup runs (1-24 hours)",
    )

    upload_file_size_limit_mb: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum file size in MB for file uploads (1-1024 MB)",
    )

    # AutoMod configuration
    automod_enabled: bool = Field(
        default=False,
        description="Whether AutoMod content moderation is enabled",
    )
    automod_api_url: str = Field(
        default="",
        description="AutoMod API base URL - Make sure it ends in /api",
    )
    automod_timeout: int = Field(
        default=30,
        description="Timeout in seconds for AutoMod API requests",
    )
    automod_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for AutoMod API requests",
    )
    automod_retry_delay: float = Field(
        default=1.0,
        description="Delay between retries for AutoMod API requests in seconds",
    )
    automod_fail_open: bool = Field(
        default=False,
        description="If True, allow execution to continue if AutoMod fails",
    )

    @field_validator("platform_base_url", "frontend_base_url")
    @classmethod
    def validate_platform_base_url(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"{info.field_name} must be a full URL "
                "including a http:// or https:// schema"
            )
        if v.endswith("/"):
            return v.rstrip("/")  # Remove trailing slash
        return v

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

    max_message_size_limit: int = Field(
        default=16 * 1024 * 1024,  # 16 MB
        description="Maximum message size limit for communication with the message bus",
    )

    backend_cors_allow_origins: List[str] = Field(default=["http://localhost:3000"])

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

    rabbitmq_default_user: str = Field(default="", description="RabbitMQ default user")
    rabbitmq_default_pass: str = Field(
        default="", description="RabbitMQ default password"
    )

    postmark_server_api_token: str = Field(
        default="", description="Postmark server API token used for sending emails"
    )

    postmark_webhook_token: str = Field(
        default="",
        description="The token to use for the Postmark webhook",
    )

    unsubscribe_secret_key: str = Field(
        default="",
        description="The secret key to use for the unsubscribe user by token",
    )

    # Cloudflare Turnstile credentials
    turnstile_secret_key: str = Field(
        default="",
        description="Cloudflare Turnstile backend secret key",
    )
    turnstile_verify_url: str = Field(
        default="https://challenges.cloudflare.com/turnstile/v0/siteverify",
        description="Cloudflare Turnstile verify URL",
    )

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
    twitter_client_id: str = Field(default="", description="Twitter/X OAuth client ID")
    twitter_client_secret: str = Field(
        default="", description="Twitter/X OAuth client secret"
    )
    discord_client_id: str = Field(default="", description="Discord OAuth client ID")
    discord_client_secret: str = Field(
        default="", description="Discord OAuth client secret"
    )

    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_internal_api_key: str = Field(
        default="", description="OpenAI Internal API key"
    )
    aiml_api_key: str = Field(default="", description="'AI/ML API' key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    open_router_api_key: str = Field(default="", description="Open Router API Key")
    llama_api_key: str = Field(default="", description="Llama API Key")
    v0_api_key: str = Field(default="", description="v0 by Vercel API key")

    reddit_client_id: str = Field(default="", description="Reddit client ID")
    reddit_client_secret: str = Field(default="", description="Reddit client secret")

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

    fal_api_key: str = Field(default="", description="FAL API key")
    exa_api_key: str = Field(default="", description="Exa API key")
    e2b_api_key: str = Field(default="", description="E2B API key")
    nvidia_api_key: str = Field(default="", description="Nvidia API key")
    mem0_api_key: str = Field(default="", description="Mem0 API key")

    linear_client_id: str = Field(default="", description="Linear client ID")
    linear_client_secret: str = Field(default="", description="Linear client secret")

    todoist_client_id: str = Field(default="", description="Todoist client ID")
    todoist_client_secret: str = Field(default="", description="Todoist client secret")

    stripe_api_key: str = Field(default="", description="Stripe API Key")
    stripe_webhook_secret: str = Field(default="", description="Stripe Webhook Secret")

    screenshotone_api_key: str = Field(default="", description="ScreenshotOne API Key")

    apollo_api_key: str = Field(default="", description="Apollo API Key")
    smartlead_api_key: str = Field(default="", description="SmartLead API Key")
    zerobounce_api_key: str = Field(default="", description="ZeroBounce API Key")
    enrichlayer_api_key: str = Field(default="", description="Enrichlayer API Key")

    # AutoMod API credentials
    automod_api_key: str = Field(default="", description="AutoMod API key")

    # LaunchDarkly feature flags
    launch_darkly_sdk_key: str = Field(
        default="",
        description="The LaunchDarkly SDK key for feature flag management",
    )

    ayrshare_api_key: str = Field(default="", description="Ayrshare API Key")
    ayrshare_jwt_key: str = Field(default="", description="Ayrshare private Key")
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
