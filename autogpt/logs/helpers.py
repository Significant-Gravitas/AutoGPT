import logging
from pathlib import Path
from typing import Any, Optional

from colorama import Fore

from .config import USER_FRIENDLY_OUTPUT_LOGGER, _chat_plugins
from .handlers import JsonFileHandler


def user_friendly_output(
    message: str,
    level: int = logging.INFO,
    title: str = "",
    title_color: str = "",
) -> None:
    """Outputs a message to the user in a user-friendly way.

    This function outputs on up to two channels:
    1. The console, in typewriter style
    2. Text To Speech, if configured
    """
    logger = logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER)

    if _chat_plugins:
        for plugin in _chat_plugins:
            plugin.report(f"{title}: {message}")

    logger.log(level, message, extra={"title": title, "title_color": title_color})


def print_attribute(
    title: str, value: Any, title_color: str = Fore.GREEN, value_color: str = ""
) -> None:
    logger = logging.getLogger()
    logger.info(
        str(value),
        extra={
            "title": f"{title.rstrip(':')}:",
            "title_color": title_color,
            "color": value_color,
        },
    )


def request_user_double_check(additionalText: Optional[str] = None) -> None:
    if not additionalText:
        additionalText = (
            "Please ensure you've setup and configured everything"
            " correctly. Read https://github.com/Torantulino/Auto-GPT#readme to "
            "double check. You can also create a github issue or join the discord"
            " and ask there!"
        )

    user_friendly_output(
        additionalText, level=logging.WARN, title="DOUBLE CHECK CONFIGURATION"
    )


def log_json(data: Any, file_name: str | Path, log_dir: Path) -> None:
    logger = logging.getLogger("JSON_LOGGER")

    # Create a handler for JSON files
    json_file_path = log_dir / file_name
    json_data_handler = JsonFileHandler(json_file_path)

    # Log the JSON data using the custom file handler
    logger.addHandler(json_data_handler)
    logger.debug(data)
    logger.removeHandler(json_data_handler)
