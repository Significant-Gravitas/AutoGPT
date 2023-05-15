"""The workspace is the central hub for the Agent's on disk resources."""
from autogpt.core.status import ShortStatus, Status
from autogpt.core.workspace.base import Workspace
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceConfiguration

status = Status(
    module_name=__name__,
    short_status=ShortStatus.BASIC_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Basic example needs to be created.\n"
        "5/11: Ported most of the existing logic and added a little extra for system setup.\n"
        "5/14: Use pydantic models for configuration.\n"
    ),
)
