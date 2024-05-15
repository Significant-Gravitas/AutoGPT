"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from colorama import Fore
from pydantic import SecretStr, validator

import autogpt
from autogpt.app.utils import clean_input
from autogpt.core.configuration.schema import (
    Configurable,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.resource.model_providers import CHAT_MODELS, ModelName
from autogpt.core.resource.model_providers.openai import (
    OpenAICredentials,
    OpenAIModelName,
)
from autogpt.file_storage import FileStorageBackendName
from autogpt.logs.config import LoggingConfig
from autogpt.speech import TTSConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(autogpt.__file__).parent.parent
AI_SETTINGS_FILE = Path("ai_settings.yaml")
AZURE_CONFIG_FILE = Path("azure.yaml")
PROMPT_SETTINGS_FILE = Path("prompt_settings.yaml")

GPT_4_MODEL = OpenAIModelName.GPT4
GPT_3_MODEL = OpenAIModelName.GPT3


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

    # TTS configuration
    logging: LoggingConfig = LoggingConfig()
    tts_config: TTSConfig = TTSConfig()

    # File storage
    file_storage_backend: FileStorageBackendName = UserConfigurable(
        default=FileStorageBackendName.LOCAL, from_env="FILE_STORAGE_BACKEND"
    )

    ##########################
    # Agent Control Settings #
    ##########################
    # Paths
    ai_settings_file: Path = UserConfigurable(
        default=AI_SETTINGS_FILE, from_env="AI_SETTINGS_FILE"
    )
    prompt_settings_file: Path = UserConfigurable(
        default=project_root / PROMPT_SETTINGS_FILE,
        from_env="PROMPT_SETTINGS_FILE",
    )

    # Model configuration
    fast_llm: ModelName = UserConfigurable(
        default=OpenAIModelName.GPT3,
        from_env="FAST_LLM",
    )
    smart_llm: ModelName = UserConfigurable(
        default=OpenAIModelName.GPT4_TURBO,
        from_env="SMART_LLM",
    )
    temperature: float = UserConfigurable(default=0, from_env="TEMPERATURE")
    openai_functions: bool = UserConfigurable(
        default=False, from_env=lambda: os.getenv("OPENAI_FUNCTIONS", "False") == "True"
    )
    embedding_model: str = UserConfigurable(
        default="text-embedding-3-small", from_env="EMBEDDING_MODEL"
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
    redis_port: int = UserConfigurable(default=6379, from_env="REDIS_PORT")
    redis_password: str = UserConfigurable("", from_env="REDIS_PASSWORD")
    wipe_redis_on_start: bool = UserConfigurable(
        default=True,
        from_env=lambda: os.getenv("WIPE_REDIS_ON_START", "True") == "True",
    )

    ############
    # Commands #
    ############
    # General
    disabled_commands: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("DISABLED_COMMANDS")),
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
    image_size: int = UserConfigurable(default=256, from_env="IMAGE_SIZE")

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

    ###############
    # Credentials #
    ###############
    # OpenAI
    openai_credentials: Optional[OpenAICredentials] = None
    azure_config_file: Optional[Path] = UserConfigurable(
        default=AZURE_CONFIG_FILE, from_env="AZURE_CONFIG_FILE"
    )

    # Github
    github_api_key: Optional[str] = UserConfigurable(from_env="GITHUB_API_KEY")
    github_username: Optional[str] = UserConfigurable(from_env="GITHUB_USERNAME")

    # Google
    google_api_key: Optional[str] = UserConfigurable(from_env="GOOGLE_API_KEY")
    google_custom_search_engine_id: Optional[str] = UserConfigurable(
        from_env="GOOGLE_CUSTOM_SEARCH_ENGINE_ID",
    )

    # Huggingface
    huggingface_api_token: Optional[str] = UserConfigurable(
        from_env="HUGGINGFACE_API_TOKEN"
    )

    # Stable Diffusion
    sd_webui_auth: Optional[str] = UserConfigurable(from_env="SD_WEBUI_AUTH")

    @validator("openai_functions")
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            assert CHAT_MODELS[smart_llm].has_function_call_api, (
                f"Model {smart_llm} does not support tool calling. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return v


class ConfigBuilder(Configurable[Config]):
    default_settings = Config()

    @classmethod
    def build_config_from_env(cls, project_root: Path = PROJECT_ROOT) -> Config:
        """Initialize the Config class"""

        config = cls.build_agent_configuration()
        config.project_root = project_root

        # Make relative paths absolute
        for k in {
            "ai_settings_file",  # TODO: deprecate or repurpose
            "prompt_settings_file",  # TODO: deprecate or repurpose
            "azure_config_file",  # TODO: move from project root
        }:
            setattr(config, k, project_root / getattr(config, k))

        if (
            config.openai_credentials
            and config.openai_credentials.api_type == "azure"
            and (config_file := config.azure_config_file)
        ):
            config.openai_credentials.load_azure_config(config_file)

        return config


def assert_config_has_openai_api_key(config: Config) -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    key_pattern = r"^sk-(proj-)?\w{48}"
    openai_api_key = (
        config.openai_credentials.api_key.get_secret_value()
        if config.openai_credentials
        else ""
    )

    # If there's no credentials or empty API key, prompt the user to set it
    if not openai_api_key:
        logger.error(
            "Please set your OpenAI API key in .env or as an environment variable."
        )
        logger.info(
            "You can get your key from https://platform.openai.com/account/api-keys"
        )
        openai_api_key = clean_input(
            config, "Please enter your OpenAI API key if you have it:"
        )
        openai_api_key = openai_api_key.strip()
        if re.search(key_pattern, openai_api_key):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            if config.openai_credentials:
                config.openai_credentials.api_key = SecretStr(openai_api_key)
            else:
                config.openai_credentials = OpenAICredentials(
                    api_key=SecretStr(openai_api_key)
                )
            print("OpenAI API key successfully set!")
            print(
                f"{Fore.YELLOW}NOTE: The API key you've set is only temporary. "
                f"For longer sessions, please set it in the .env file{Fore.RESET}"
            )
        else:
            print(f"{Fore.RED}Invalid OpenAI API key{Fore.RESET}")
            exit(1)
    # If key is set, but it looks invalid
    elif not re.search(key_pattern, openai_api_key):
        logger.error(
            "Invalid OpenAI API key! "
            "Please set your OpenAI API key in .env or as an environment variable."
        )
        logger.info(
            "You can get your key from https://platform.openai.com/account/api-keys"
        )
        exit(1)


def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)
