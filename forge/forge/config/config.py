"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import SecretStr, validator

import forge
from forge.file_storage import FileStorageBackendName
from forge.llm.providers import CHAT_MODELS, ModelName
from forge.llm.providers.openai import OpenAICredentials, OpenAIModelName
from forge.logging.config import LoggingConfig
from forge.models.config import Configurable, SystemSettings, UserConfigurable
from forge.speech.say import TTSConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(forge.__file__).parent.parent
AZURE_CONFIG_FILE = Path("azure.yaml")

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
            "azure_config_file",  # TODO: move from project root
        }:
            setattr(config, k, project_root / getattr(config, k))

        if (
            config.openai_credentials
            and config.openai_credentials.api_type == SecretStr("azure")
            and (config_file := config.azure_config_file)
        ):
            config.openai_credentials.load_azure_config(config_file)

        return config


async def assert_config_has_required_llm_api_keys(config: Config) -> None:
    """
    Check if API keys (if required) are set for the configured SMART_LLM and FAST_LLM.
    """
    from pydantic import ValidationError

    from forge.llm.providers.anthropic import AnthropicModelName
    from forge.llm.providers.groq import GroqModelName

    if set((config.smart_llm, config.fast_llm)).intersection(AnthropicModelName):
        from forge.llm.providers.anthropic import AnthropicCredentials

        try:
            credentials = AnthropicCredentials.from_env()
        except ValidationError as e:
            if "api_key" in str(e):
                logger.error(
                    "Set your Anthropic API key in .env or as an environment variable"
                )
                logger.info(
                    "For further instructions: "
                    "https://docs.agpt.co/autogpt/setup/#anthropic"
                )

            raise ValueError("Anthropic is unavailable: can't load credentials") from e

        key_pattern = r"^sk-ant-api03-[\w\-]{95}"

        # If key is set, but it looks invalid
        if not re.search(key_pattern, credentials.api_key.get_secret_value()):
            logger.warning(
                "Possibly invalid Anthropic API key! "
                f"Configured Anthropic API key does not match pattern '{key_pattern}'. "
                "If this is a valid key, please report this warning to the maintainers."
            )

    if set((config.smart_llm, config.fast_llm)).intersection(GroqModelName):
        from groq import AuthenticationError

        from forge.llm.providers.groq import GroqProvider

        try:
            groq = GroqProvider()
            await groq.get_available_models()
        except ValidationError as e:
            if "api_key" not in str(e):
                raise

            logger.error("Set your Groq API key in .env or as an environment variable")
            logger.info(
                "For further instructions: https://docs.agpt.co/autogpt/setup/#groq"
            )
            raise ValueError("Groq is unavailable: can't load credentials")
        except AuthenticationError as e:
            logger.error("The Groq API key is invalid!")
            logger.info(
                "For instructions to get and set a new API key: "
                "https://docs.agpt.co/autogpt/setup/#groq"
            )
            raise ValueError("Groq is unavailable: invalid API key") from e

    if set((config.smart_llm, config.fast_llm)).intersection(OpenAIModelName):
        from openai import AuthenticationError

        from forge.llm.providers.openai import OpenAIProvider

        try:
            openai = OpenAIProvider()
            await openai.get_available_models()
        except ValidationError as e:
            if "api_key" not in str(e):
                raise

            logger.error(
                "Set your OpenAI API key in .env or as an environment variable"
            )
            logger.info(
                "For further instructions: https://docs.agpt.co/autogpt/setup/#openai"
            )
            raise ValueError("OpenAI is unavailable: can't load credentials")
        except AuthenticationError as e:
            logger.error("The OpenAI API key is invalid!")
            logger.info(
                "For instructions to get and set a new API key: "
                "https://docs.agpt.co/autogpt/setup/#openai"
            )
            raise ValueError("OpenAI is unavailable: invalid API key") from e


def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)
