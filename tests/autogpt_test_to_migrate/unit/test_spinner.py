import time

from autogpt.app.spinner import Spinner

ALMOST_DONE_MESSAGE = "Almost done..."
PLEASE_WAIT = "Please wait..."


def test_spinner_initializes_with_default_values():
    """Tests that the spinner initializes with default values."""
    with Spinner() as spinner:
        assert spinner.message == "Loading..."
        assert spinner.delay == 0.1


def test_spinner_initializes_with_custom_values():
    """Tests that the spinner initializes with custom message and delay values."""
    with Spinner(message=PLEASE_WAIT, delay=0.2) as spinner:
        assert spinner.message == PLEASE_WAIT
        assert spinner.delay == 0.2


#
def test_spinner_stops_spinning():
    """Tests that the spinner starts spinning and stops spinning without errors."""
    with Spinner() as spinner:
        time.sleep(1)
    assert not spinner.running


def test_spinner_can_be_used_as_context_manager():
    """Tests that the spinner can be used as a context manager."""
    with Spinner() as spinner:
        assert spinner.running
    assert not spinner.running
