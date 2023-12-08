import logging


def get_client_logger(logger_level: int = logging.DEBUG):
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    client_logger = logging.getLogger("autogpt_client_application")
    client_logger.setLevel(logger_level)

    formatter = logging.Formatter(
        "%(asctime)s#%(filename)s:%(funcName)s:%(levelname)s:%(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logger_level)
    ch.setFormatter(formatter)

    client_logger.addHandler(ch)

    return client_logger
