"""The workspace is the central hub for the Agent's on disk resources."""
from autogpt.core.workspace.base import Workspace

import autogpt.core.status

status = autogpt.core.status.Status.INTERFACE_DONE
handover_notes = "Interface has been created. Basic example needs to be created."
