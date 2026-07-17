"""Configuration class to store the state of bools for different scripts access."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import SecretStr

from forge.config.base import BaseConfig
from forge.llm.providers import ModelName
from forge.llm.providers.openai import OpenAICredentials, OpenAIModelName
from forge.logging.config import LoggingConfig
from forge.models.config import Configurable, UserConfigurable

# Type alias for prompt strategy options
PromptStrategyName = Literal[
    "one_shot",
    "rewoo",
    "plan_execute",
    "reflexion",
    "tree_of_thoughts",
    "lats",
    "multi_agent_debate",
]

logger = logging.getLogger(__name__)

AZURE_CONFIG_FILE = Path("azure.yaml")

GPT_4_MODEL = OpenAIModelName.GPT4_O
GPT_3_MODEL = (
    OpenAIModelName.GPT4_O_MINI
)  # Fallback model for when configured model is unavailable


class AppConfig(BaseConfig):
    name: str = "Auto-GPT configuration"
    description: str = "Default configuration for the Auto-GPT application."

    ########################
    # Application Settings #
    ########################
    workspace: Path = Path.cwd()
    app_data_dir: Path = workspace / ".autogpt"
    skip_news: bool = False
    skip_reprompt: bool = False
    authorise_key: str = UserConfigurable(default="y", from_env="AUTHORISE_COMMAND_KEY")
    exit_key: str = UserConfigurable(default="n", from_env="EXIT_KEY")
    noninteractive_mode: bool = UserConfigurable(
        default=False, from_env="NONINTERACTIVE_MODE"
    )
    logging: LoggingConfig = LoggingConfig()
    component_config_file: Optional[Path] = UserConfigurable(
        default=None, from_env="COMPONENT_CONFIG_FILE"
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
    embedding_model: str = UserConfigurable(
        default="text-embedding-3-small", from_env="EMBEDDING_MODEL"
    )
    thinking_budget_tokens: Optional[int] = UserConfigurable(
        default=None, from_env="THINKING_BUDGET_TOKENS"
    )
    """Token budget for extended thinking (Anthropic Claude models). Minimum 1024."""
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = UserConfigurable(
        default=None, from_env="REASONING_EFFORT"
    )
    """Reasoning effort level for OpenAI o-series and GPT-5 models."""

    # Run loop configuration
    continuous_mode: bool = True
    continuous_limit: int = 0

    # Prompt strategy configuration
    prompt_strategy: PromptStrategyName = UserConfigurable(
        default="one_shot",
        from_env="PROMPT_STRATEGY",
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

    ###############
    # Credentials #
    ###############
    # OpenAI
    openai_credentials: Optional[OpenAICredentials] = None
    azure_config_file: Optional[Path] = UserConfigurable(
        default=AZURE_CONFIG_FILE, from_env="AZURE_CONFIG_FILE"
    )


class ConfigBuilder(Configurable[AppConfig]):
    default_settings = AppConfig()

    @classmethod
    def build_config_from_env(cls, workspace: Optional[Path] = None) -> AppConfig:
        """Initialize the Config class

        Args:
            workspace: The workspace directory where AutoGPT will operate.
                       Defaults to current working directory.
        """
        if workspace is None:
            workspace = Path.cwd()

        config = cls.build_agent_configuration()
        config.workspace = workspace
        config.app_data_dir = workspace / ".autogpt"

        # Make relative paths absolute
        for k in {
            "azure_config_file",
        }:
            setattr(config, k, workspace / getattr(config, k))

        if (
            config.openai_credentials
            and config.openai_credentials.api_type == SecretStr("azure")
            and (config_file := config.azure_config_file)
        ):
            config.openai_credentials.load_azure_config(config_file)

        return config


async def assert_config_has_required_llm_api_keys(config: AppConfig) -> None:
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
                    "https://docs.agpt.co/classic/original_autogpt/setup/#anthropic"
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
                "For further instructions: "
                + "https://docs.agpt.co/classic/original_autogpt/setup/#groq"
            )
            raise ValueError("Groq is unavailable: can't load credentials")
        except AuthenticationError as e:
            logger.error("The Groq API key is invalid!")
            logger.info(
                "For instructions to get and set a new API key: "
                "https://docs.agpt.co/classic/original_autogpt/setup/#groq"
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
                "For further instructions: "
                + "https://docs.agpt.co/classic/original_autogpt/setup/#openai"
            )
            raise ValueError("OpenAI is unavailable: can't load credentials")
        except AuthenticationError as e:
            logger.error("The OpenAI API key is invalid!")
            logger.info(
                "For instructions to get and set a new API key: "
                "https://docs.agpt.co/classic/original_autogpt/setup/#openai"
            )
            raise ValueError("OpenAI is unavailable: invalid API key") from e


def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)
