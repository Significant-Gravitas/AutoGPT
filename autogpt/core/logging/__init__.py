"""The logging system allows the Agent to record its activities."""
from autogpt.core.logging.base import Logger
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes="Interface has been created. Basic example needs to be created.",
)
