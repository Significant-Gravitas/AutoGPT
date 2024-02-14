"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore
from pydantic import Field, SecretStr, validator

import autogpt
from autogpt.core.configuration.schema import (
    Configurable,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAICredentials,
)
from autogpt.file_workspace import FileWorkspaceBackendName
from autogpt.logs.config import LoggingConfig
from autogpt.plugins.plugins_config import PluginsConfig
from autogpt.speech import TTSConfig

PROJECT_ROOT = Path(autogpt.__file__).parent.parent
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

    # TTS configuration
    tts_config: TTSConfig = TTSConfig()
    logging: LoggingConfig = LoggingConfig()

    # Workspace
    workspace_backend: FileWorkspaceBackendName = UserConfigurable(
        default=FileWorkspaceBackendName.LOCAL,
        from_env=lambda: FileWorkspaceBackendName(v)
        if (v := os.getenv("WORKSPACE_BACKEND"))
        else None,
    )

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
        default="gpt-3.5-turbo-0125",
        from_env=lambda: os.getenv("FAST_LLM"),
    )
    smart_llm: str = UserConfigurable(
        default="gpt-4-turbo-preview",
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
    plugins_config: PluginsConfig = Field(
        default_factory=lambda: PluginsConfig(plugins={})
    )
    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)
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
    openai_credentials: Optional[OpenAICredentials] = None
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

    # White/Black Listing URLs
    web_policy: Optional[str] = os.getenv("WEB_POLICY", 0)
    url_list: Optional[list] = os.getenv("URL_LIST",[]).split(",")

    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    @validator("openai_functions")
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            assert OPEN_AI_CHAT_MODELS[smart_llm].has_function_call_api, (
                f"Model {smart_llm} does not support OpenAI Functions. "
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
            "plugins_config_file",  # TODO: move from project root
            "azure_config_file",  # TODO: move from project root
        }:
            setattr(config, k, project_root / getattr(config, k))

        if (
            config.openai_credentials
            and config.openai_credentials.api_type == "azure"
            and (config_file := config.azure_config_file)
        ):
            config.openai_credentials.load_azure_config(config_file)

        config.plugins_config = PluginsConfig.load_config(
            config.plugins_config_file,
            config.plugins_denylist,
            config.plugins_allowlist,
        )

        return config


def assert_config_has_openai_api_key(config: Config) -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not config.openai_credentials:
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
            + Fore.RESET
        )
        print("You can get your key from https://platform.openai.com/account/api-keys")
        openai_api_key = input(
            "If you do have the key, please enter your OpenAI API key now:\n"
        )
        key_pattern = r"^sk-\w{48}"
        openai_api_key = openai_api_key.strip()
        if re.search(key_pattern, openai_api_key):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            config.openai_credentials = OpenAICredentials(
                api_key=SecretStr(openai_api_key)
            )
            print(
                Fore.GREEN
                + "OpenAI API key successfully set!\n"
                + Fore.YELLOW
                + "NOTE: The API key you've set is only temporary.\n"
                + "For longer sessions, please set it in .env file"
                + Fore.RESET
            )
        else:
            print("Invalid OpenAI API key!")
            exit(1)


def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)
