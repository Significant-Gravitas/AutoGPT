import sys

from autogpt.exceptions import HandledException


def handle_exception(exc_type, exc_value, exc_traceback):
    """A custom exception handler that manages HandledException"""
    if issubclass(exc_type, HandledException):
        sys.stderr.write(f"{exc_value.message}\n")
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


def register_exception_handler() -> None:
    """Register the custom exception handler"""
    sys.excepthook = handle_exception
