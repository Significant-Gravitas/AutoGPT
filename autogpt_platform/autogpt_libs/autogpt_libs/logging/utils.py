import logging
import re
from typing import Any

import uvicorn.config
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


def generate_uvicorn_config():
    """
    Generates a uvicorn logging config that silences uvicorn's default logging and tells it to use the native logging module.
    """
    log_config = dict(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"]["uvicorn"] = {"handlers": []}
    log_config["loggers"]["uvicorn.error"] = {"handlers": []}
    log_config["loggers"]["uvicorn.access"] = {"handlers": []}
    return log_config
