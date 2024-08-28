def configure_logging():
    import logging

    from forge.logging import configure_logging

    configure_logging()

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
