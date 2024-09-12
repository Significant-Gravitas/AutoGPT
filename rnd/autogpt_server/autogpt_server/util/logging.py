import os

from autogpt_libs.logging.config import configure_logging


def configure_logging():
    import logging

    from autogpt_libs.logging import configure_logging

    if os.getenv("APP_ENV") != "cloud":
        configure_logging()
    else:
        configure_logging(force_cloud_logging=True)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
