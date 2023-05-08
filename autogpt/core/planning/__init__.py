"""The planning system organizes the Agent's activities."""
from autogpt.core.planning.base import Planner

import autogpt.core.status

status = autogpt.core.status.Status.INTERFACE_DONE
handover_notes = "Interface has been created. Basic example needs to be created."
