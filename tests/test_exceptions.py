import sys

from autogpt.exceptions import (
    CriticalException,
    HandledException,
    register_exception_handler,
)
from autogpt.exceptions.exception_handler import handle_exception


def test_handle_exception_handled_exception(capsys):
    # Simulate a HandledException
    exception = HandledException("Handled Exception Message")

    # Call the handle_exception function
    handle_exception(type(exception), exception, None)

    # Check if the exception message is printed to stderr
    captured = capsys.readouterr()
    assert captured.err == "Handled Exception Message\n"


def test_handle_exception_unhandled_exception(capsys):
    # Simulate an unhandled exception
    exception = ValueError("Unhandled Exception Message")

    # Call the handle_exception function
    handle_exception(type(exception), exception, None)

    # Check if the exception is propagated to the default exception handler
    captured = capsys.readouterr()
    assert captured.err == "ValueError: Unhandled Exception Message\n"


def test_register_exception_handler():
    # Call the register_exception_handler function
    register_exception_handler()

    # Check if sys.excepthook is set to handle_exception
    assert sys.excepthook == handle_exception


def test_handled_exception():
    # Create a HandledException
    exception = HandledException("Handled Exception Message")

    # Verify the exception message
    assert exception.message == "Handled Exception Message"


def test_critical_exception():
    # Create a CriticalException
    exception = CriticalException("Critical Exception Message")

    # Verify the exception message
    assert exception.message == "Critical Exception Message"
