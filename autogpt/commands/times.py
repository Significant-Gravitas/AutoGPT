from datetime import datetime

from autogpt.commands.command import command


@command(
    "get_times",
    "Get Times",
)
def get_datetime() -> str:
    """Return the current date and time

    Returns:
        str: The current date and time
    """
    return "Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
