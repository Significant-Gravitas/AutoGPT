"""The logging system allows the Agent to record its activities."""
from autogpt.core.logging.base import Logger

import autogpt.core.status

status = autogpt.core.status.Status.INTERFACE_DONE
handover_notes = "Interface has been created. Basic example needs to be created."