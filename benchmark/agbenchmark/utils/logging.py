from __future__ import annotations

import logging

from colorama import Fore, Style

SIMPLE_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
DEBUG_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)03d  %(message)s"


def configure_logging(
    level: int = logging.INFO,
) -> None:
    """Configure the native logging module."""

    # Auto-adjust default log format based on log level
    log_format = DEBUG_LOG_FORMAT if level == logging.DEBUG else SIMPLE_LOG_FORMAT

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FancyConsoleFormatter(log_format))

    # Configure the root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[console_handler],
    )


class FancyConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter designed for console output.

    This formatter enhances the standard logging output with color coding. The color
    coding is based on the level of the log message, making it easier to distinguish
    between different types of messages in the console output.

    The color for each level is defined in the LEVEL_COLOR_MAP class attribute.
    """

    # level -> (level & text color, title color)
    LEVEL_COLOR_MAP = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Make sure `msg` is a string
        if not hasattr(record, "msg"):
            record.msg = ""
        elif not type(record.msg) is str:
            record.msg = str(record.msg)

        # Justify the level name to 5 characters minimum
        record.levelname = record.levelname.ljust(5)

        # Determine default color based on error level
        level_color = ""
        if record.levelno in self.LEVEL_COLOR_MAP:
            level_color = self.LEVEL_COLOR_MAP[record.levelno]
            record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        # Determine color for message
        color = getattr(record, "color", level_color)
        color_is_specified = hasattr(record, "color")

        # Don't color INFO messages unless the color is explicitly specified.
        if color and (record.levelno != logging.INFO or color_is_specified):
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        return super().format(record)
