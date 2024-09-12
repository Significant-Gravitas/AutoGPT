import logging
import os

import autogpt_libs.logging.config


def configure_logging():

    if os.getenv("APP_ENV") != "cloud":
        autogpt_libs.logging.config.configure_logging()
    else:
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=True)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
