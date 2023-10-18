"""Configurator module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import click
from colorama import Back, Fore, Style

from autogpt import utils
from autogpt.config import Config
from autogpt.config.config import GPT_3_MODEL, GPT_4_MODEL
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.helpers import print_attribute, request_user_double_check
from autogpt.memory.vector import get_supported_memory_backends

logger = logging.getLogger(__name__)


def apply_overrides_to_config(
    config: Config,
    continuous: bool = False,
    continuous_limit: Optional[int] = None,
    ai_settings_file: Optional[Path] = None,
    prompt_settings_file: Optional[Path] = None,
    skip_reprompt: bool = False,
    speak: bool = False,
    debug: bool = False,
    gpt3only: bool = False,
    gpt4only: bool = False,
    memory_type: str = "",
    browser_name: str = "",
    allow_downloads: bool = False,
    skip_news: bool = False,
) -> None:
    """Updates the config object with the given arguments.

    Args:
        continuous (bool): Whether to run in continuous mode
        continuous_limit (int): The number of times to run in continuous mode
        ai_settings_file (Path): The path to the ai_settings.yaml file
        prompt_settings_file (Path): The path to the prompt_settings.yaml file
        skip_reprompt (bool): Whether to skip the re-prompting messages at the beginning of the script
        speak (bool): Whether to enable speak mode
        debug (bool): Whether to enable debug mode
        gpt3only (bool): Whether to enable GPT3.5 only mode
        gpt4only (bool): Whether to enable GPT4 only mode
        memory_type (str): The type of memory backend to use
        browser_name (str): The name of the browser to use when using selenium to scrape the web
        allow_downloads (bool): Whether to allow AutoGPT to download files natively
        skips_news (bool): Whether to suppress the output of latest news on startup
    """
    config.debug_mode = False
    config.continuous_mode = False
    config.tts_config.speak_mode = False

    if debug:
        print_attribute("Debug mode", "ENABLED")
        config.debug_mode = True

    if continuous:
        print_attribute("Continuous Mode", "ENABLED", title_color=Fore.YELLOW)
        logger.warning(
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        config.continuous_mode = True

        if continuous_limit:
            print_attribute("Continuous Limit", continuous_limit)
            config.continuous_limit = continuous_limit

    # Check if continuous limit is used without continuous mode
    if continuous_limit and not continuous:
        raise click.UsageError("--continuous-limit can only be used with --continuous")

    if speak:
        print_attribute("Speak Mode", "ENABLED")
        config.tts_config.speak_mode = True

    # Set the default LLM models
    if gpt3only:
        print_attribute("GPT3.5 Only Mode", "ENABLED")
        # --gpt3only should always use gpt-3.5-turbo, despite user's FAST_LLM config
        config.fast_llm = GPT_3_MODEL
        config.smart_llm = GPT_3_MODEL
    elif (
        gpt4only
        and check_model(GPT_4_MODEL, model_type="smart_llm", config=config)
        == GPT_4_MODEL
    ):
        print_attribute("GPT4 Only Mode", "ENABLED")
        # --gpt4only should always use gpt-4, despite user's SMART_LLM config
        config.fast_llm = GPT_4_MODEL
        config.smart_llm = GPT_4_MODEL
    else:
        config.fast_llm = check_model(config.fast_llm, "fast_llm", config=config)
        config.smart_llm = check_model(config.smart_llm, "smart_llm", config=config)

    if memory_type:
        supported_memory = get_supported_memory_backends()
        chosen = memory_type
        if chosen not in supported_memory:
            logger.warning(
                extra={
                    "title": "ONLY THE FOLLOWING MEMORY BACKENDS ARE SUPPORTED:",
                    "title_color": Fore.RED,
                },
                msg=f"{supported_memory}",
            )
            print_attribute(
                "Defaulting to", config.memory_backend, title_color=Fore.YELLOW
            )
        else:
            config.memory_backend = chosen

    if skip_reprompt:
        print_attribute("Skip Re-prompt", "ENABLED")
        config.skip_reprompt = True

    if ai_settings_file:
        file = ai_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.fatal(extra={"title": "FAILED FILE VALIDATION:"}, msg=message)
            request_user_double_check()
            exit(1)

        print_attribute("Using AI Settings File", file)
        config.ai_settings_file = config.project_root / file
        config.skip_reprompt = True

    if prompt_settings_file:
        file = prompt_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.fatal(extra={"title": "FAILED FILE VALIDATION:"}, msg=message)
            request_user_double_check()
            exit(1)

        print_attribute("Using Prompt Settings File", file)
        config.prompt_settings_file = config.project_root / file

    if browser_name:
        config.selenium_web_browser = browser_name

    if allow_downloads:
        print_attribute("Native Downloading", "ENABLED")
        logger.warn(
            msg=f"{Back.LIGHTYELLOW_EX}AutoGPT will now be able to download and save files to your machine.{Back.RESET}"
            " It is recommended that you monitor any files it downloads carefully.",
        )
        logger.warn(
            msg=f"{Back.RED + Style.BRIGHT}ALWAYS REMEMBER TO NEVER OPEN FILES YOU AREN'T SURE OF!{Style.RESET_ALL}",
        )
        config.allow_downloads = True

    if skip_news:
        config.skip_news = True


def check_model(
    model_name: str,
    model_type: Literal["smart_llm", "fast_llm"],
    config: Config,
) -> str:
    """Check if model is available for use. If not, return gpt-3.5-turbo."""
    openai_credentials = config.get_openai_credentials(model_name)
    api_manager = ApiManager()
    models = api_manager.get_models(**openai_credentials)

    if any(model_name in m["id"] for m in models):
        return model_name

    logger.warn(
        f"You do not have access to {model_name}. Setting {model_type} to gpt-3.5-turbo."
    )
    return "gpt-3.5-turbo"
