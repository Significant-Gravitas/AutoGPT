import logging


def get_client_logger():
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    client_logger = logging.getLogger("autogpt_client_application")
    client_logger.setLevel(logging.DEBUG)
    return client_logger
