import logging
from typing import Any, Optional

from colorama import Fore

from .config import SPEECH_OUTPUT_LOGGER, USER_FRIENDLY_OUTPUT_LOGGER


def user_friendly_output(
    message: str,
    level: int = logging.INFO,
    title: str = "",
    title_color: str = "",
    preserve_message_color: bool = False,
) -> None:
    """Outputs a message to the user in a user-friendly way.

    This function outputs on up to two channels:
    1. The console, in typewriter style
    2. Text To Speech, if configured
    """
    logger = logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER)

    logger.log(
        level,
        message,
        extra={
            "title": title,
            "title_color": title_color,
            "preserve_color": preserve_message_color,
        },
    )


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
            "Please ensure you've setup and configured everything correctly. "
            "Read https://docs.agpt.co/autogpt/setup/ to double check. "
            "You can also create a github issue or join the discord and ask there!"
        )

    user_friendly_output(
        additionalText,
        level=logging.WARN,
        title="DOUBLE CHECK CONFIGURATION",
        preserve_message_color=True,
    )


def speak(message: str, level: int = logging.INFO) -> None:
    logging.getLogger(SPEECH_OUTPUT_LOGGER).log(level, message)
