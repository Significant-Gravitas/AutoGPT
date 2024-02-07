"""Configuration class to store the state of bools for different scripts access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field, field_validator

import AFAAS
from AFAAS.configs.schema import Configurable, SystemSettings, UserConfigurable

# # from AFAAS.core.adapters.openai.chatmodel import OPEN_AI_CHAT_MODELS
# from AFAAS.interfaces.adapters.language_model import BaseModelProviderCredentials


PROJECT_ROOT = Path(str(AFAAS.__path__)).parent
AI_SETTINGS_FILE = Path("ai_settings.yaml")
AZURE_CONFIG_FILE = Path("azure.yaml")
PLUGINS_CONFIG_FILE = Path("plugins_config.yaml")
PROMPT_SETTINGS_FILE = Path("prompt_settings.yaml")

GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"


class Config(SystemSettings, arbitrary_types_allowed=True):
    name: str = "Auto-GPT configuration"
    description: str = "Default configuration for the Auto-GPT application."

    ########################
    # Application Settings #
    ########################
    project_root: Path = PROJECT_ROOT
    app_data_dir: Path = project_root / "data"
    skip_news: bool = False
    skip_reprompt: bool = False
    authorise_key: str = UserConfigurable(default="y", from_env="AUTHORISE_COMMAND_KEY")
    exit_key: str = UserConfigurable(default="n", from_env="EXIT_KEY")
    noninteractive_mode: bool = False
    chat_messages_enabled: bool = UserConfigurable(
        default=True, from_env=lambda: os.getenv("CHAT_MESSAGES_ENABLED") == "True"
    )

    # # TTS configuration
    # tts_config: TTSConfig = TTSConfig()
    # logging: LoggingConfig = LoggingConfig()

    # # Workspace
    # workspace_backend: AbstractFileWorkspaceBackendName = UserConfigurable(
    #     default=FileWorkspaceBackendName.LOCAL,
    #     from_env=lambda: AbstractFileWorkspaceBackendName(v)
    #     if (v := os.getenv("WORKSPACE_BACKEND"))
    #     else None,
    # )

    ##########################
    # Agent Control Settings #
    ##########################
    # Paths
    ai_settings_file: Path = UserConfigurable(
        default=AI_SETTINGS_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("AI_SETTINGS_FILE")) else None,
    )
    prompt_settings_file: Path = UserConfigurable(
        default=PROMPT_SETTINGS_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("PROMPT_SETTINGS_FILE")) else None,
    )

    # Model configuration
    fast_llm: str = UserConfigurable(
        default="gpt-3.5-turbo-16k",
        from_env=lambda: os.getenv("FAST_LLM"),
    )
    smart_llm: str = UserConfigurable(
        default="gpt-4",
        from_env=lambda: os.getenv("SMART_LLM"),
    )
    temperature: float = UserConfigurable(
        default=0,
        from_env=lambda: float(v) if (v := os.getenv("TEMPERATURE")) else None,
    )
    openai_functions: bool = UserConfigurable(
        default=False, from_env=lambda: os.getenv("OPENAI_FUNCTIONS", "False") == "True"
    )
    embedding_model: str = UserConfigurable(
        default="text-embedding-ada-002", from_env="EMBEDDING_MODEL"
    )
    browse_spacy_language_model: str = UserConfigurable(
        default="en_core_web_sm", from_env="BROWSE_SPACY_LANGUAGE_MODEL"
    )

    # Run loop configuration
    continuous_mode: bool = False
    continuous_limit: int = 0

    ##########
    # Memory #
    ##########
    memory_backend: str = UserConfigurable("json_file", from_env="MEMORY_BACKEND")
    memory_index: str = UserConfigurable("auto-gpt-memory", from_env="MEMORY_INDEX")
    redis_host: str = UserConfigurable("localhost", from_env="REDIS_HOST")
    redis_port: int = UserConfigurable(
        default=6379,
        from_env=lambda: int(v) if (v := os.getenv("REDIS_PORT")) else None,
    )
    redis_password: str = UserConfigurable("", from_env="REDIS_PASSWORD")
    wipe_redis_on_start: bool = UserConfigurable(
        default=True,
        from_env=lambda: os.getenv("WIPE_REDIS_ON_START", "True") == "True",
    )

    ############
    # Commands #
    ############
    # General
    disabled_command_categories: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("DISABLED_COMMAND_CATEGORIES")),
    )

    # File ops
    restrict_to_workspace: bool = UserConfigurable(
        default=True,
        from_env=lambda: os.getenv("RESTRICT_TO_WORKSPACE", "True") == "True",
    )
    allow_downloads: bool = False

    # Shell commands
    shell_command_control: str = UserConfigurable(
        default="denylist", from_env="SHELL_COMMAND_CONTROL"
    )
    execute_local_commands: bool = UserConfigurable(
        default=False,
        from_env=lambda: os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True",
    )
    shell_denylist: list[str] = UserConfigurable(
        default_factory=lambda: ["sudo", "su"],
        from_env=lambda: _safe_split(
            os.getenv("SHELL_DENYLIST", os.getenv("DENY_COMMANDS"))
        ),
    )
    shell_allowlist: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(
            os.getenv("SHELL_ALLOWLIST", os.getenv("ALLOW_COMMANDS"))
        ),
    )

    # Text to image
    image_provider: Optional[str] = UserConfigurable(from_env="IMAGE_PROVIDER")
    huggingface_image_model: str = UserConfigurable(
        default="CompVis/stable-diffusion-v1-4", from_env="HUGGINGFACE_IMAGE_MODEL"
    )
    sd_webui_url: Optional[str] = UserConfigurable(
        default="http://localhost:7860", from_env="SD_WEBUI_URL"
    )
    image_size: int = UserConfigurable(
        default=256,
        from_env=lambda: int(v) if (v := os.getenv("IMAGE_SIZE")) else None,
    )

    # Audio to text
    audio_to_text_provider: str = UserConfigurable(
        default="huggingface", from_env="AUDIO_TO_TEXT_PROVIDER"
    )
    huggingface_audio_to_text_model: Optional[str] = UserConfigurable(
        from_env="HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
    )

    # Web browsing
    selenium_web_browser: str = UserConfigurable("chrome", from_env="USE_WEB_BROWSER")
    selenium_headless: bool = UserConfigurable(
        default=True, from_env=lambda: os.getenv("HEADLESS_BROWSER", "True") == "True"
    )
    user_agent: str = UserConfigurable(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",  # noqa: E501
        from_env="USER_AGENT",
    )

    ###################
    # Plugin Settings #
    ###################
    plugins_dir: str = UserConfigurable("plugins", from_env="PLUGINS_DIR")
    plugins_config_file: Path = UserConfigurable(
        default=PLUGINS_CONFIG_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("PLUGINS_CONFIG_FILE")) else None,
    )
    # plugins_config: PluginsConfig = Field(
    #     default_factory=lambda: PluginsConfig(plugins={})
    # )
    plugins: list = Field(default_factory=list, exclude=True)
    plugins_allowlist: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("ALLOWLISTED_PLUGINS")),
    )
    plugins_denylist: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("DENYLISTED_PLUGINS")),
    )
    plugins_openai: list[str] = UserConfigurable(
        default_factory=list, from_env=lambda: _safe_split(os.getenv("OPENAI_PLUGINS"))
    )

    ###############
    # Credentials #
    ###############
    # OpenAI
    # openai_credentials: Optional[BaseModelProviderCredentials] = None
    azure_config_file: Optional[Path] = UserConfigurable(
        default=AZURE_CONFIG_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("AZURE_CONFIG_FILE")) else None,
    )

    # Github
    github_api_key: Optional[str] = UserConfigurable(from_env="GITHUB_API_KEY")
    github_username: Optional[str] = UserConfigurable(from_env="GITHUB_USERNAME")

    # Google
    google_api_key: Optional[str] = UserConfigurable(from_env="GOOGLE_API_KEY")
    google_custom_search_engine_id: Optional[str] = UserConfigurable(
        from_env=lambda: os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID"),
    )

    # Huggingface
    huggingface_api_token: Optional[str] = UserConfigurable(
        from_env="HUGGINGFACE_API_TOKEN"
    )

    # Stable Diffusion
    sd_webui_auth: Optional[str] = UserConfigurable(from_env="SD_WEBUI_AUTH")


def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)
