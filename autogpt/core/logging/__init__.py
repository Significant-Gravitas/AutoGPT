"""The logging system allows the Agent to record its activities."""
from autogpt.core.logging.base import Logger
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Basic example needs to be created.\n"
        "5/8: Thinking we should deprecate this entirely and just use std logging."
    ),
)
