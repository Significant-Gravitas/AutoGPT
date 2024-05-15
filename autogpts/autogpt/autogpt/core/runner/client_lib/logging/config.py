import logging
import sys

from forge.logging import BelowLevelFilter, FancyConsoleFormatter
from openai._base_client import log as openai_logger

SIMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s  %(message)s"
DEBUG_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d  %(message)s"
)


def configure_root_logger():
    console_formatter = FancyConsoleFormatter(SIMPLE_LOG_FORMAT)

    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.DEBUG)
    stdout.addFilter(BelowLevelFilter(logging.WARNING))
    stdout.setFormatter(console_formatter)
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(console_formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[stdout, stderr])

    # Disable debug logging from OpenAI library
    openai_logger.setLevel(logging.WARNING)
