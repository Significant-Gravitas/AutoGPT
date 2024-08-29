from forge.logging.config import LogFormatName


def configure_logging():
    import logging

    from forge.logging import configure_logging

    configure_logging(log_format=LogFormatName.STRUCTURED)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
