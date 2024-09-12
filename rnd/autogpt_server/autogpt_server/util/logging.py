import os


def configure_logging():
    import logging

    import autogpt_libs.logging.config

    if os.getenv("APP_ENV") != "cloud":
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=False)
    else:
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=True)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
