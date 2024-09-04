import os

from forge.logging.config import LogFormatName


def configure_logging():
    import logging

    from forge.logging import configure_logging

    if os.getenv("APP_ENV") != "cloud":
        configure_logging()
    else:
        configure_logging(log_format=LogFormatName.STRUCTURED)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
