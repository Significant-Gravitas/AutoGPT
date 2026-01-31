"""Introspect Pydantic models to extract UserConfigurable fields."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal, Type, Union, get_args, get_origin

from pydantic import BaseModel, SecretStr
from pydantic.fields import FieldInfo

from forge.models.config import _get_field_metadata


@dataclass
class SettingInfo:
    """Information about a configurable setting."""

    name: str
    env_var: str
    description: str
    field_type: str  # "str", "int", "float", "bool", "secret", "choice"
    choices: list[str] = field(default_factory=list)
    default: Any = None
    required: bool = False

    def get_display_value(self, value: Any) -> str:
        """Get display-friendly representation of a value."""
        if value is None:
            return "[not set]"
        if self.field_type == "secret":
            secret_val = (
                value.get_secret_value() if isinstance(value, SecretStr) else str(value)
            )
            if not secret_val:
                return "[not set]"
            # Mask all but first 3 and last 4 characters
            if len(secret_val) > 10:
                return f"{secret_val[:3]}...{secret_val[-4:]}"
            return "***"
        if self.field_type == "bool":
            return "true" if value else "false"
        return str(value)


def _extract_field_type(field_info: FieldInfo) -> tuple[str, list[str]]:
    """Extract the field type and choices from a Pydantic FieldInfo.

    Returns:
        Tuple of (field_type, choices) where field_type is one of:
        "str", "int", "float", "bool", "secret", "choice"
    """
    annotation = field_info.annotation
    choices: list[str] = []

    # Unwrap Optional
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # Filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            annotation = non_none_args[0]
            origin = get_origin(annotation)

    # Check for SecretStr
    if annotation is SecretStr:
        return "secret", []

    # Check for Literal (choices)
    if origin is Literal:
        choices = list(get_args(annotation))
        return "choice", [str(c) for c in choices]

    # Check for Enum
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        choices = [e.value for e in annotation]
        return "choice", choices

    # Check basic types
    if annotation is bool:
        return "bool", []
    if annotation is int:
        return "int", []
    if annotation is float:
        return "float", []
    if annotation is str:
        return "str", []

    # Default to string
    return "str", []


def extract_configurable_fields(
    model_class: Type[BaseModel],
) -> list[SettingInfo]:
    """Extract all UserConfigurable fields from a Pydantic model.

    Args:
        model_class: A Pydantic BaseModel class

    Returns:
        List of SettingInfo objects for each configurable field
    """
    settings: list[SettingInfo] = []

    for name, field_info in model_class.model_fields.items():
        # Check if this field is user configurable
        if not _get_field_metadata(field_info, "user_configurable"):
            continue

        # Get the environment variable name
        from_env = _get_field_metadata(field_info, "from_env")
        if from_env is None:
            continue

        # Handle callable from_env (skip these - they're complex)
        if callable(from_env):
            continue

        env_var = from_env
        field_type, choices = _extract_field_type(field_info)

        # Get default value
        default = field_info.default
        if default is not None and hasattr(default, "__class__"):
            # Handle PydanticUndefined
            if "PydanticUndefined" in str(type(default)):
                default = None

        settings.append(
            SettingInfo(
                name=name,
                env_var=env_var,
                description=field_info.description or "",
                field_type=field_type,
                choices=choices,
                default=default,
                required=field_info.is_required(),
            )
        )

    return settings


def get_all_configurable_settings() -> dict[str, SettingInfo]:
    """Get all configurable settings from known models.

    Returns:
        Dict mapping environment variable names to SettingInfo
    """
    from autogpt.app.config import AppConfig

    from forge.llm.providers.anthropic import AnthropicCredentials
    from forge.llm.providers.groq import GroqCredentials
    from forge.llm.providers.openai import OpenAICredentials
    from forge.logging.config import LoggingConfig

    settings: dict[str, SettingInfo] = {}

    # Extract from all known models
    models = [
        AppConfig,
        OpenAICredentials,
        AnthropicCredentials,
        GroqCredentials,
        LoggingConfig,
    ]

    for model in models:
        for setting in extract_configurable_fields(model):
            # Use env_var as key to deduplicate
            if setting.env_var not in settings:
                settings[setting.env_var] = setting

    return settings


# Additional env vars from .env.template that aren't in models
ADDITIONAL_ENV_VARS: dict[str, SettingInfo] = {
    "TAVILY_API_KEY": SettingInfo(
        name="tavily_api_key",
        env_var="TAVILY_API_KEY",
        description="Tavily API key for AI-optimized search",
        field_type="secret",
    ),
    "SERPER_API_KEY": SettingInfo(
        name="serper_api_key",
        env_var="SERPER_API_KEY",
        description="Serper.dev API key for Google SERP results",
        field_type="secret",
    ),
    "GOOGLE_API_KEY": SettingInfo(
        name="google_api_key",
        env_var="GOOGLE_API_KEY",
        description="Google API key (deprecated, use Serper)",
        field_type="secret",
    ),
    "GOOGLE_CUSTOM_SEARCH_ENGINE_ID": SettingInfo(
        name="google_cse_id",
        env_var="GOOGLE_CUSTOM_SEARCH_ENGINE_ID",
        description="Google Custom Search Engine ID (deprecated)",
        field_type="str",
    ),
    "HUGGINGFACE_API_TOKEN": SettingInfo(
        name="huggingface_api_token",
        env_var="HUGGINGFACE_API_TOKEN",
        description="HuggingFace API token for image generation",
        field_type="secret",
    ),
    "SD_WEBUI_AUTH": SettingInfo(
        name="sd_webui_auth",
        env_var="SD_WEBUI_AUTH",
        description="Stable Diffusion Web UI username:password",
        field_type="secret",
    ),
    "GITHUB_API_KEY": SettingInfo(
        name="github_api_key",
        env_var="GITHUB_API_KEY",
        description="GitHub API key / PAT",
        field_type="secret",
    ),
    "GITHUB_USERNAME": SettingInfo(
        name="github_username",
        env_var="GITHUB_USERNAME",
        description="GitHub username",
        field_type="str",
    ),
    "TEXT_TO_SPEECH_PROVIDER": SettingInfo(
        name="tts_provider",
        env_var="TEXT_TO_SPEECH_PROVIDER",
        description="Text-to-speech provider",
        field_type="choice",
        choices=["gtts", "streamelements", "elevenlabs", "macos"],
        default="gtts",
    ),
    "ELEVENLABS_API_KEY": SettingInfo(
        name="elevenlabs_api_key",
        env_var="ELEVENLABS_API_KEY",
        description="Eleven Labs API key",
        field_type="secret",
    ),
    "ELEVENLABS_VOICE_ID": SettingInfo(
        name="elevenlabs_voice_id",
        env_var="ELEVENLABS_VOICE_ID",
        description="Eleven Labs voice ID",
        field_type="str",
    ),
    "STREAMELEMENTS_VOICE": SettingInfo(
        name="streamelements_voice",
        env_var="STREAMELEMENTS_VOICE",
        description="StreamElements voice name",
        field_type="str",
        default="Brian",
    ),
    "FILE_STORAGE_BACKEND": SettingInfo(
        name="file_storage_backend",
        env_var="FILE_STORAGE_BACKEND",
        description="Storage backend for file operations",
        field_type="choice",
        choices=["local", "gcs", "s3"],
        default="local",
    ),
    "STORAGE_BUCKET": SettingInfo(
        name="storage_bucket",
        env_var="STORAGE_BUCKET",
        description="GCS/S3 bucket name",
        field_type="str",
    ),
    "S3_ENDPOINT_URL": SettingInfo(
        name="s3_endpoint_url",
        env_var="S3_ENDPOINT_URL",
        description="S3 endpoint URL (for non-AWS S3)",
        field_type="str",
    ),
    "AP_SERVER_PORT": SettingInfo(
        name="ap_server_port",
        env_var="AP_SERVER_PORT",
        description="Agent Protocol server port",
        field_type="int",
        default=8000,
    ),
    "AP_SERVER_DB_URL": SettingInfo(
        name="ap_server_db_url",
        env_var="AP_SERVER_DB_URL",
        description="Agent Protocol database URL",
        field_type="str",
    ),
    "AP_SERVER_CORS_ALLOWED_ORIGINS": SettingInfo(
        name="ap_server_cors_origins",
        env_var="AP_SERVER_CORS_ALLOWED_ORIGINS",
        description="CORS allowed origins (comma-separated)",
        field_type="str",
    ),
    "TELEMETRY_OPT_IN": SettingInfo(
        name="telemetry_opt_in",
        env_var="TELEMETRY_OPT_IN",
        description="Share telemetry with AutoGPT team",
        field_type="bool",
        default=False,
    ),
    "PLAIN_OUTPUT": SettingInfo(
        name="plain_output",
        env_var="PLAIN_OUTPUT",
        description="Disable animated typing and spinner",
        field_type="bool",
        default=False,
    ),
    # Platform integration
    "PLATFORM_API_KEY": SettingInfo(
        name="platform_api_key",
        env_var="PLATFORM_API_KEY",
        description="AutoGPT Platform API key for blocks integration",
        field_type="secret",
    ),
    "PLATFORM_BLOCKS_ENABLED": SettingInfo(
        name="platform_blocks_enabled",
        env_var="PLATFORM_BLOCKS_ENABLED",
        description="Enable platform blocks integration",
        field_type="bool",
        default=True,
    ),
    "PLATFORM_URL": SettingInfo(
        name="platform_url",
        env_var="PLATFORM_URL",
        description="AutoGPT Platform URL",
        field_type="str",
        default="https://platform.agpt.co",
    ),
    "PLATFORM_TIMEOUT": SettingInfo(
        name="platform_timeout",
        env_var="PLATFORM_TIMEOUT",
        description="Platform API timeout in seconds",
        field_type="int",
        default=60,
    ),
    # Groq settings
    "GROQ_API_BASE_URL": SettingInfo(
        name="groq_api_base_url",
        env_var="GROQ_API_BASE_URL",
        description="Groq API base URL (for custom endpoints)",
        field_type="str",
    ),
    # Llamafile settings
    "LLAMAFILE_API_BASE": SettingInfo(
        name="llamafile_api_base",
        env_var="LLAMAFILE_API_BASE",
        description="Llamafile API base URL",
        field_type="str",
        default="http://localhost:8080/v1",
    ),
}


def get_complete_settings() -> dict[str, SettingInfo]:
    """Get all settings including additional env vars."""
    settings = get_all_configurable_settings()
    settings.update(ADDITIONAL_ENV_VARS)
    return settings
