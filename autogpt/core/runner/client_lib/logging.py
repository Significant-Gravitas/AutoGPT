import logging


def get_client_logger():
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    client_logger = logging.getLogger("autogpt_client_application")
    client_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    client_logger.addHandler(ch)

    return client_logger
