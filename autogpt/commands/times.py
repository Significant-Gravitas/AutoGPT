import time
from datetime import datetime


def get_datetime() -> str:
    """Return the current date and time

    Returns:
        str: The current date and time
    """
    return "Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wait_seconds(seconds: int) -> str:
    """Halts the overall execution for a set amount of seconds

    Returns:
        str: A confirmation that the wait has been executed
    """
    time.sleep(int(seconds))
    return f"Waited for {seconds} seconds."
