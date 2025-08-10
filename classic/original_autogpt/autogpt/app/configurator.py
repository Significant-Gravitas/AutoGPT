"""Configurator module."""
from __future__ import annotations

import logging
from typing import Literal, Optional

import click
from forge.llm.providers import ModelName, MultiProvider

from autogpt.app.config import GPT_3_MODEL, AppConfig

logger = logging.getLogger(__name__)


async def apply_overrides_to_config(
    config: AppConfig,
    continuous: bool = False,
    continuous_limit: Optional[int] = None,
    skip_reprompt: bool = False,
    skip_news: bool = False,
) -> None:
    """Updates the config object with the given arguments.

    Args:
        config (Config): The config object to update.
        continuous (bool): Whether to run in continuous mode.
        continuous_limit (int): The number of times to run in continuous mode.
        skip_reprompt (bool): Whether to skip the re-prompting messages on start.
        speak (bool): Whether to enable speak mode.
        debug (bool): Whether to enable debug mode.
        log_level (int): The global log level for the application.
        log_format (str): The format for the log(s).
        log_file_format (str): Override the format for the log file.
        skips_news (bool): Whether to suppress the output of latest news on startup.
    """
    config.continuous_mode = False

    if continuous:
        logger.warning(
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        config.continuous_mode = True

        if continuous_limit:
            config.continuous_limit = continuous_limit

    # Check if continuous limit is used without continuous mode
    if continuous_limit and not continuous:
        raise click.UsageError("--continuous-limit can only be used with --continuous")

    # Check availability of configured LLMs; fallback to other LLM if unavailable
    config.fast_llm, config.smart_llm = await check_models(
        (config.fast_llm, "fast_llm"), (config.smart_llm, "smart_llm")
    )

    if skip_reprompt:
        config.skip_reprompt = True

    if skip_news:
        config.skip_news = True


async def check_models(
    *models: tuple[ModelName, Literal["smart_llm", "fast_llm"]]
) -> tuple[ModelName, ...]:
    """Check if model is available for use. If not, return gpt-3.5-turbo."""
    multi_provider = MultiProvider()
    available_models = await multi_provider.get_available_chat_models()

    checked_models: list[ModelName] = []
    for model, model_type in models:
        if any(model == m.name for m in available_models):
            checked_models.append(model)
        else:
            logger.warning(
                f"You don't have access to {model}. "
                f"Setting {model_type} to {GPT_3_MODEL}."
            )
            checked_models.append(GPT_3_MODEL)

    return tuple(checked_models)
