"""Logging module for Auto-GPT."""

import logging
import sys
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from .filters import BelowLevelFilter
from .formatters import AGPTFormatter, StructuredLoggingFormatter

LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOG_FILE = "activity.log"
DEBUG_LOG_FILE = "debug.log"
ERROR_LOG_FILE = "error.log"

SIMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s  %(title)s%(message)s"

DEBUG_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(filename)s:%(lineno)d" "  %(title)s%(message)s"
)


class LoggingConfig(BaseSettings):

    level: str = Field(
        default="INFO",
        description="Logging level",
        validation_alias="LOG_LEVEL",
    )

    enable_cloud_logging: bool = Field(
        default=False,
        description="Enable logging to Google Cloud Logging",
    )

    enable_file_logging: bool = Field(
        default=False,
        description="Enable logging to file",
    )
    # File output
    log_dir: Path = Field(
        default=LOG_DIR,
        description="Log directory",
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("level", mode="before")
    @classmethod
    def parse_log_level(cls, v):
        if isinstance(v, str):
            v = v.upper()
            if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Invalid log level: {v}")
            return v
        return v


def configure_logging(force_cloud_logging: bool = False) -> None:
    """Configure the native logging module based on the LoggingConfig settings.

    This function sets up logging handlers and formatters according to the
    configuration specified in the LoggingConfig object. It supports various
    logging outputs including console, file, cloud, and JSON logging.

    The function uses the LoggingConfig object to determine which logging
    features to enable and how to configure them. This includes setting
    log levels, log formats, and output destinations.

    No arguments are required as the function creates its own LoggingConfig
    instance internally.

    Note: This function is typically called at the start of the application
    to set up the logging infrastructure.
    """

    config = LoggingConfig()

    log_handlers: list[logging.Handler] = []

    # Cloud logging setup
    if config.enable_cloud_logging or force_cloud_logging:
        import google.cloud.logging
        from google.cloud.logging.handlers import CloudLoggingHandler
        from google.cloud.logging_v2.handlers.transports.sync import SyncTransport

        client = google.cloud.logging.Client()
        cloud_handler = CloudLoggingHandler(
            client,
            name="autogpt_logs",
            transport=SyncTransport,
        )
        cloud_handler.setLevel(config.level)
        cloud_handler.setFormatter(StructuredLoggingFormatter())
        log_handlers.append(cloud_handler)
        print("Cloud logging enabled")
    else:
        # Console output handlers
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(config.level)
        stdout.addFilter(BelowLevelFilter(logging.WARNING))
        if config.level == logging.DEBUG:
            stdout.setFormatter(AGPTFormatter(DEBUG_LOG_FORMAT))
        else:
            stdout.setFormatter(AGPTFormatter(SIMPLE_LOG_FORMAT))

        stderr = logging.StreamHandler()
        stderr.setLevel(logging.WARNING)
        if config.level == logging.DEBUG:
            stderr.setFormatter(AGPTFormatter(DEBUG_LOG_FORMAT))
        else:
            stderr.setFormatter(AGPTFormatter(SIMPLE_LOG_FORMAT))

        log_handlers += [stdout, stderr]
        print("Console logging enabled")

    # File logging setup
    if config.enable_file_logging:
        # create log directory if it doesn't exist
        if not config.log_dir.exists():
            config.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Log directory: {config.log_dir}")

        # Activity log handler (INFO and above)
        activity_log_handler = logging.FileHandler(
            config.log_dir / LOG_FILE, "a", "utf-8"
        )
        activity_log_handler.setLevel(config.level)
        activity_log_handler.setFormatter(
            AGPTFormatter(SIMPLE_LOG_FORMAT, no_color=True)
        )
        log_handlers.append(activity_log_handler)

        if config.level == logging.DEBUG:
            # Debug log handler (all levels)
            debug_log_handler = logging.FileHandler(
                config.log_dir / DEBUG_LOG_FILE, "a", "utf-8"
            )
            debug_log_handler.setLevel(logging.DEBUG)
            debug_log_handler.setFormatter(
                AGPTFormatter(DEBUG_LOG_FORMAT, no_color=True)
            )
            log_handlers.append(debug_log_handler)

        # Error log handler (ERROR and above)
        error_log_handler = logging.FileHandler(
            config.log_dir / ERROR_LOG_FILE, "a", "utf-8"
        )
        error_log_handler.setLevel(logging.ERROR)
        error_log_handler.setFormatter(AGPTFormatter(DEBUG_LOG_FORMAT, no_color=True))
        log_handlers.append(error_log_handler)
        print("File logging enabled")

    # Configure the root logger
    logging.basicConfig(
        format=DEBUG_LOG_FORMAT if config.level == logging.DEBUG else SIMPLE_LOG_FORMAT,
        level=config.level,
        handlers=log_handlers,
    )
