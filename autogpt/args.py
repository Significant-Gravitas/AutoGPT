"""This module contains the argument parsing logic for the script."""
import argparse

from colorama import Fore
from autogpt import utils
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.memory import get_supported_memory_backends

CFG = Config()


def parse_arguments() -> None:
    """Parses the arguments passed to the script

    Returns:
        None
    """
    CFG.set_debug_mode(False)
    CFG.set_continuous_mode(False)
    CFG.set_speak_mode(False)

    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument(
        "--continuous", "-c", action="store_true", help="Enable Continuous Mode"
    )
    parser.add_argument(
        "--continuous-limit",
        "-l",
        type=int,
        dest="continuous_limit",
        help="Defines the number of times to run in continuous mode",
    )
    parser.add_argument("--speak", action="store_true", help="Enable Speak Mode")
    parser.add_argument("--debug", action="store_true", help="Enable Debug Mode")
    parser.add_argument(
        "--gpt3only", action="store_true", help="Enable GPT3.5 Only Mode"
    )
    parser.add_argument("--gpt4only", action="store_true", help="Enable GPT4 Only Mode")
    parser.add_argument(
        "--use-memory",
        "-m",
        dest="memory_type",
        help="Defines which Memory backend to use",
    )
    parser.add_argument(
        "--skip-reprompt",
        "-y",
        dest="skip_reprompt",
        action="store_true",
        help="Skips the re-prompting messages at the beginning of the script",
    )
    parser.add_argument(
        "--use-browser",
        "-b",
        dest="browser_name",
        help="Specifies which web-browser to use when using selenium to scrape the web.",
    )
    parser.add_argument(
        "--ai-settings",
        "-C",
        dest="ai_settings_file",
        help="Specifies which ai_settings.yaml file to use, will also automatically"
        " skip the re-prompt.",
    )
    args = parser.parse_args()

    if args.debug:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_debug_mode(True)

    if args.continuous:
        logger.typewriter_log("Continuous Mode: ", Fore.RED, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        CFG.set_continuous_mode(True)

        if args.continuous_limit:
            logger.typewriter_log(
                "Continuous Limit: ", Fore.GREEN, f"{args.continuous_limit}"
            )
            CFG.set_continuous_limit(args.continuous_limit)

    # Check if continuous limit is used without continuous mode
    if args.continuous_limit and not args.continuous:
        parser.error("--continuous-limit can only be used with --continuous")

    if args.speak:
        logger.typewriter_log("Speak Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_speak_mode(True)

    if args.gpt3only:
        logger.typewriter_log("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_smart_llm_model(CFG.fast_llm_model)

    if args.gpt4only:
        logger.typewriter_log("GPT4 Only Mode: ", Fore.GREEN, "ENABLED")
        CFG.set_fast_llm_model(CFG.smart_llm_model)

    if args.memory_type:
        supported_memory = get_supported_memory_backends()
        chosen = args.memory_type
        if chosen not in supported_memory:
            logger.typewriter_log(
                "ONLY THE FOLLOWING MEMORY BACKENDS ARE SUPPORTED: ",
                Fore.RED,
                f"{supported_memory}",
            )
            logger.typewriter_log("Defaulting to: ", Fore.YELLOW, CFG.memory_backend)
        else:
            CFG.memory_backend = chosen

    if args.skip_reprompt:
        logger.typewriter_log("Skip Re-prompt: ", Fore.GREEN, "ENABLED")
        CFG.skip_reprompt = True

    if args.ai_settings_file:
        file = args.ai_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.typewriter_log("FAILED FILE VALIDATION", Fore.RED, message)
            logger.double_check()
            exit(1)

        logger.typewriter_log("Using AI Settings File:", Fore.GREEN, file)
        CFG.ai_settings_file = file
        CFG.skip_reprompt = True

    if args.browser_name:
        CFG.selenium_web_browser = args.browser_name
