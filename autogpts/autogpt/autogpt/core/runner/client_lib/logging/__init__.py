import logging

from .config import BelowLevelFilter, FancyConsoleFormatter, configure_root_logger
from .helpers import dump_prompt


def get_client_logger(logger_level = logging.DEBUG):
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    client_logger = logging.getLogger("autogpt_client_application")
    client_logger.setLevel(logger_level)

    return client_logger


__all__ = [
    "configure_root_logger",
    "get_client_logger",
    "FancyConsoleFormatter",
    "BelowLevelFilter",
    "dump_prompt",
]