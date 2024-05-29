"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import click
import forge
from colorama import Fore
from forge.config.base import BaseConfig
from forge.llm.providers import CHAT_MODELS, ModelName
from forge.llm.providers.openai import OpenAICredentials, OpenAIModelName
from forge.logging.config import LoggingConfig
from forge.models.config import Configurable, UserConfigurable
from pydantic import SecretStr, validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(forge.__file__).parent.parent
AZURE_CONFIG_FILE = Path("azure.yaml")

GPT_4_MODEL = OpenAIModelName.GPT4
GPT_3_MODEL = OpenAIModelName.GPT3


class Config(BaseConfig):
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
    logging: LoggingConfig = LoggingConfig()

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

    ###############
    # Credentials #
    ###############
    # OpenAI
    openai_credentials: Optional[OpenAICredentials] = None
    azure_config_file: Optional[Path] = UserConfigurable(
        default=AZURE_CONFIG_FILE, from_env="AZURE_CONFIG_FILE"
    )

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
        openai_api_key = click.prompt(
            "Please enter your OpenAI API key if you have it",
            default="",
            show_default=False,
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
