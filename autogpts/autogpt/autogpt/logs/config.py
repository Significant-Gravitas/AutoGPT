"""Logging module for Auto-GPT."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from openai.util import logger as openai_logger

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.core.runner.client_lib.logging import BelowLevelFilter

from .formatters import AutoGptFormatter
from .handlers import TTSHandler, TypingConsoleHandler

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = "activity.log"
DEBUG_LOG_FILE = "debug.log"
ERROR_LOG_FILE = "error.log"

SIMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s  %(title)s%(message)s"
DEBUG_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d"
    "  %(title)s%(message)s"
)

SPEECH_OUTPUT_LOGGER = "VOICE"
USER_FRIENDLY_OUTPUT_LOGGER = "USER_FRIENDLY_OUTPUT"

_chat_plugins: list[AutoGPTPluginTemplate] = []


def configure_logging(config: Config, log_dir: Path = LOG_DIR) -> None:
    """Configure the native logging module."""

    # create log directory if it doesn't exist
    if not log_dir.exists():
        log_dir.mkdir()

    log_level = logging.DEBUG if config.debug_mode else logging.INFO
    log_format = DEBUG_LOG_FORMAT if config.debug_mode else SIMPLE_LOG_FORMAT
    console_formatter = AutoGptFormatter(log_format)

    # Console output handlers
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(log_level)
    stdout.addFilter(BelowLevelFilter(logging.WARNING))
    stdout.setFormatter(console_formatter)
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(console_formatter)

    # INFO log file handler
    activity_log_handler = logging.FileHandler(log_dir / LOG_FILE, "a", "utf-8")
    activity_log_handler.setLevel(logging.INFO)
    activity_log_handler.setFormatter(AutoGptFormatter(SIMPLE_LOG_FORMAT))

    if config.debug_mode:
        # DEBUG log file handler
        debug_log_handler = logging.FileHandler(log_dir / DEBUG_LOG_FILE, "a", "utf-8")
        debug_log_handler.setLevel(logging.DEBUG)
        debug_log_handler.setFormatter(AutoGptFormatter(DEBUG_LOG_FORMAT))

    # ERROR log file handler
    error_log_handler = logging.FileHandler(log_dir / ERROR_LOG_FILE, "a", "utf-8")
    error_log_handler.setLevel(logging.ERROR)
    error_log_handler.setFormatter(AutoGptFormatter(DEBUG_LOG_FORMAT))

    # Configure the root logger
    logging.basicConfig(
        format=log_format,
        level=log_level,
        handlers=(
            [stdout, stderr, activity_log_handler, error_log_handler]
            + ([debug_log_handler] if config.debug_mode else [])
        ),
    )

    ## Set up user-friendly loggers

    # Console output handler which simulates typing
    typing_console_handler = TypingConsoleHandler(stream=sys.stdout)
    typing_console_handler.setLevel(logging.INFO)
    typing_console_handler.setFormatter(console_formatter)

    user_friendly_output_logger = logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER)
    user_friendly_output_logger.setLevel(logging.INFO)
    user_friendly_output_logger.addHandler(
        typing_console_handler if not config.plain_output else stdout
    )
    user_friendly_output_logger.addHandler(TTSHandler(config))
    user_friendly_output_logger.addHandler(activity_log_handler)
    user_friendly_output_logger.addHandler(error_log_handler)
    user_friendly_output_logger.addHandler(stderr)
    user_friendly_output_logger.propagate = False

    speech_output_logger = logging.getLogger(SPEECH_OUTPUT_LOGGER)
    speech_output_logger.setLevel(logging.INFO)
    speech_output_logger.addHandler(TTSHandler(config))
    speech_output_logger.propagate = False

    # JSON logger with better formatting
    json_logger = logging.getLogger("JSON_LOGGER")
    json_logger.setLevel(logging.DEBUG)
    json_logger.propagate = False

    # Disable debug logging from OpenAI library
    openai_logger.setLevel(logging.INFO)


def configure_chat_plugins(config: Config) -> None:
    """Configure chat plugins for use by the logging module"""

    logger = logging.getLogger(__name__)

    # Add chat plugins capable of report to logger
    if config.chat_messages_enabled:
        if _chat_plugins:
            _chat_plugins.clear()

        for plugin in config.plugins:
            if hasattr(plugin, "can_handle_report") and plugin.can_handle_report():
                logger.debug(f"Loaded plugin into logger: {plugin.__class__.__name__}")
                _chat_plugins.append(plugin)
