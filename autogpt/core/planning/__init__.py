"""The planning system organizes the Agent's activities."""
from autogpt.core.planning.base import Planner
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes="Interface has been created. Basic example needs to be created.",
)
