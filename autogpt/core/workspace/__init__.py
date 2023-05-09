"""The workspace is the central hub for the Agent's on disk resources."""
from autogpt.core.status import ShortStatus, Status
from autogpt.core.workspace.base import Workspace

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Basic example needs to be created."
    ),
)
