from datetime import datetime

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


def get_datetime() -> str:
    """Return the current date and time

    Returns:
        str: The current date and time
    """
    LOG.warning(
        "We recommand against using this tool & include the current data & time in your prompt"
    )
    return "Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
