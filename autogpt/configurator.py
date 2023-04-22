"""Configurator module."""
import click
from colorama import Back, Fore, Style

from autogpt import utils
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.memory import get_supported_memory_backends

CFG = Config()


def create_config(
    continuous: bool,
    continuous_limit: int,
    ai_settings_file: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    global_kill_switch: str,
    global_kill_switch_canary: str,
    local_kill_switch: str,
    local_kill_switch_canary: str,
) -> None:
    """Updates the config object with the given arguments.

    Args:
        continuous (bool): Whether to run in continuous mode
        continuous_limit (int): The number of times to run in continuous mode
        ai_settings_file (str): The path to the ai_settings.yaml file
        skip_reprompt (bool): Whether to skip the re-prompting messages at the beginning of the script
        speak (bool): Whether to enable speak mode
        debug (bool): Whether to enable debug mode
        gpt3only (bool): Whether to enable GPT3.5 only mode
        gpt4only (bool): Whether to enable GPT4 only mode
        memory_type (str): The type of memory backend to use
        browser_name (str): The name of the browser to use when using selenium to scrape the web
        allow_downloads (bool): Whether to allow Auto-GPT to download files natively
        skips_news (bool): Whether to suppress the output of latest news on startup
        global_kill_switch (str): The URL to check for the global kill switch canary
        global_kill_switch_canary (str): The string to check for on the global kill switch web page
        local_kill_switch (str): The URL to check for the local kill switch canary
        local_kill_switch_canary (str): The string to check for on the local kill switch web page
    """
    CFG.set_debug_mode(False)
    CFG.set_continuous_mode(False)
    CFG.set_speak_mode(False)

    if debug:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_debug_mode(True)

    if continuous:
        logger.typewriter_log("Continuous Mode: ", Fore.RED, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        # Provide an extra warning when continuous mode is used unlimited with no kill switch
        if (
            not continuous_limit
            and not global_kill_switch
            and not local_kill_switch
        ):
            logger.typewriter_log(
                "WARNING: ",
                Fore.RED,
                "Continuous mode is being launched without a kill switch.",
            )

        CFG.set_continuous_mode(True)

        if continuous_limit:
            logger.typewriter_log(
                "Continuous Limit: ", Fore.GREEN, f"{continuous_limit}"
            )
            CFG.set_continuous_limit(continuous_limit)

    # Check if continuous limit is used without continuous mode
    if continuous_limit and not continuous:
        raise click.UsageError("--continuous-limit can only be used with --continuous")

    if speak:
        logger.typewriter_log("Speak Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_speak_mode(True)

    if gpt3only:
        logger.typewriter_log("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_smart_llm_model(CFG.fast_llm_model)

    if gpt4only:
        logger.typewriter_log("GPT4 Only Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_fast_llm_model(CFG.smart_llm_model)

    if memory_type:
        supported_memory = get_supported_memory_backends()
        chosen = memory_type
        if chosen not in supported_memory:
            logger.typewriter_log(
                "ONLY THE FOLLOWING MEMORY BACKENDS ARE SUPPORTED: ",
                Fore.RED,
                f"{supported_memory}",
            )
            logger.typewriter_log("Defaulting to: ", Fore.YELLOW, CFG.memory_backend)
        else:
            CFG.memory_backend = chosen

    if skip_reprompt:
        logger.typewriter_log("Skip Re-prompt: ", Fore.GREEN, "ENABLED")
        CFG.skip_reprompt = True

    if global_kill_switch:
        CFG.global_kill_switch = global_kill_switch
    if global_kill_switch_canary:
        CFG.global_kill_switch_canary = global_kill_switch_canary
    if local_kill_switch:
        CFG.local_kill_switch = local_kill_switch
    if local_kill_switch_canary:
        CFG.local_kill_switch_canary = local_kill_switch_canary

    if ai_settings_file:
        file = ai_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.typewriter_log("FAILED FILE VALIDATION", Fore.RED, message)
            logger.double_check()
            exit(1)

        logger.typewriter_log("Using AI Settings File:", Fore.GREEN, file)
        CFG.ai_settings_file = file
        CFG.skip_reprompt = True

    if browser_name:
        CFG.selenium_web_browser = browser_name

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
        CFG.allow_downloads = True

    if skip_news:
        CFG.skip_news = True
