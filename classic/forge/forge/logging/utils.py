import logging
import re
from typing import Any

from colorama import Fore


def remove_color_codes(s: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", s)


def fmt_kwargs(kwargs: dict) -> str:
    return ", ".join(f"{n}={repr(v)}" for n, v in kwargs.items())


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


def speak(message: str, level: int = logging.INFO) -> None:
    from .config import SPEECH_OUTPUT_LOGGER

    logging.getLogger(SPEECH_OUTPUT_LOGGER).log(level, message)
