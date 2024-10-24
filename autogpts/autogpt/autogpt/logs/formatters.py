import logging

from colorama import Style
from google.cloud.logging_v2.handlers import CloudLoggingFilter, StructuredLogHandler

from autogpt.core.runner.client_lib.logging import FancyConsoleFormatter

from .utils import remove_color_codes


class AutoGptFormatter(FancyConsoleFormatter):
    def __init__(self, *args, no_color: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_color = no_color

    def format(self, record: logging.LogRecord) -> str:
        # Make sure `msg` is a string
        if not hasattr(record, "msg"):
            record.msg = ""
        elif not type(record.msg) is str:
            record.msg = str(record.msg)

        # Strip color from the message to prevent color spoofing
        if record.msg and not getattr(record, "preserve_color", False):
            record.msg = remove_color_codes(record.msg)

        # Determine color for title
        title = getattr(record, "title", "")
        title_color = getattr(record, "title_color", "") or self.LEVEL_COLOR_MAP.get(
            record.levelno, ""
        )
        if title and title_color:
            title = f"{title_color + Style.BRIGHT}{title}{Style.RESET_ALL}"
        # Make sure record.title is set, and padded with a space if not empty
        record.title = f"{title} " if title else ""

        if self.no_color:
            return remove_color_codes(super().format(record))
        else:
            return super().format(record)


class StructuredLoggingFormatter(StructuredLogHandler, logging.Formatter):
    def __init__(self):
        # Set up CloudLoggingFilter to add diagnostic info to the log records
        self.cloud_logging_filter = CloudLoggingFilter()

        # Init StructuredLogHandler
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        self.cloud_logging_filter.filter(record)
        return super().format(record)
