"""Configurator module."""
from __future__ import annotations

from typing import TYPE_CHECKING

import click
from colorama import Back, Fore, Style

from autogpt import utils
from autogpt.llm.utils import check_model
from autogpt.logs import logger
from autogpt.memory.vector import get_supported_memory_backends

if TYPE_CHECKING:
    from autogpt.config import Config

GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"


def create_config(
    config: Config,
    continuous: bool,
    continuous_limit: int,
    ai_settings_file: str,
    prompt_settings_file: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
) -> None:
    """Updates the config object with the given arguments.

    Args:
        continuous (bool): Whether to run in continuous mode
        continuous_limit (int): The number of times to run in continuous mode
        ai_settings_file (str): The path to the ai_settings.yaml file
        prompt_settings_file (str): The path to the prompt_settings.yaml file
        skip_reprompt (bool): Whether to skip the re-prompting messages at the beginning of the script
        speak (bool): Whether to enable speak mode
        debug (bool): Whether to enable debug mode
        gpt3only (bool): Whether to enable GPT3.5 only mode
        gpt4only (bool): Whether to enable GPT4 only mode
        memory_type (str): The type of memory backend to use
        browser_name (str): The name of the browser to use when using selenium to scrape the web
        allow_downloads (bool): Whether to allow Auto-GPT to download files natively
        skips_news (bool): Whether to suppress the output of latest news on startup
    """
    config.set_debug_mode(False)
    config.set_continuous_mode(False)
    config.set_speak_mode(False)

    if debug:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")
        config.set_debug_mode(True)

    if continuous:
        logger.typewriter_log("Continuous Mode: ", Fore.RED, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        config.set_continuous_mode(True)

        if continuous_limit:
            logger.typewriter_log(
                "Continuous Limit: ", Fore.GREEN, f"{continuous_limit}"
            )
            config.set_continuous_limit(continuous_limit)

    # Check if continuous limit is used without continuous mode
    if continuous_limit and not continuous:
        raise click.UsageError("--continuous-limit can only be used with --continuous")

    if speak:
        logger.typewriter_log("Speak Mode: ", Fore.GREEN, "ENABLED")
        config.set_speak_mode(True)

    # Set the default LLM models
    if gpt3only:
        logger.typewriter_log("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        # --gpt3only should always use gpt-3.5-turbo, despite user's FAST_LLM_MODEL config
        config.set_fast_llm_model(GPT_3_MODEL)
        config.set_smart_llm_model(GPT_3_MODEL)

    elif (
        gpt4only
        and check_model(GPT_4_MODEL, model_type="smart_llm_model") == GPT_4_MODEL
    ):
        logger.typewriter_log("GPT4 Only Mode: ", Fore.GREEN, "ENABLED")
        # --gpt4only should always use gpt-4, despite user's SMART_LLM_MODEL config
        config.set_fast_llm_model(GPT_4_MODEL)
        config.set_smart_llm_model(GPT_4_MODEL)
    else:
        config.set_fast_llm_model(check_model(config.fast_llm_model, "fast_llm_model"))
        config.set_smart_llm_model(
            check_model(config.smart_llm_model, "smart_llm_model")
        )

    if memory_type:
        supported_memory = get_supported_memory_backends()
        chosen = memory_type
        if chosen not in supported_memory:
            logger.typewriter_log(
                "ONLY THE FOLLOWING MEMORY BACKENDS ARE SUPPORTED: ",
                Fore.RED,
                f"{supported_memory}",
            )
            logger.typewriter_log("Defaulting to: ", Fore.YELLOW, config.memory_backend)
        else:
            config.memory_backend = chosen

    if skip_reprompt:
        logger.typewriter_log("Skip Re-prompt: ", Fore.GREEN, "ENABLED")
        config.skip_reprompt = True

    if ai_settings_file:
        file = ai_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.typewriter_log("FAILED FILE VALIDATION", Fore.RED, message)
            logger.double_check()
            exit(1)

        logger.typewriter_log("Using AI Settings File:", Fore.GREEN, file)
        config.ai_settings_file = file
        config.skip_reprompt = True

    if prompt_settings_file:
        file = prompt_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.typewriter_log("FAILED FILE VALIDATION", Fore.RED, message)
            logger.double_check()
            exit(1)

        logger.typewriter_log("Using Prompt Settings File:", Fore.GREEN, file)
        config.prompt_settings_file = file

    if browser_name:
        config.selenium_web_browser = browser_name

    if allow_downloads:
        logger.typewriter_log("Native Downloading:", Fore.GREEN, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.YELLOW,
            f"{Back.LIGHTYELLOW_EX}Auto-GPT will now be able to download and save files to your machine.{Back.RESET} "
            + "It is recommended that you monitor any files it downloads carefully.",
        )
        logger.typewriter_log(
            "WARNING: ",
            Fore.YELLOW,
            f"{Back.RED + Style.BRIGHT}ALWAYS REMEMBER TO NEVER OPEN FILES YOU AREN'T SURE OF!{Style.RESET_ALL}",
        )
        config.allow_downloads = True

    if skip_news:
        config.skip_news = True
