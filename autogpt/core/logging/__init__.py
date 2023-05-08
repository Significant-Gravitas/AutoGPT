"""The logging system allows the Agent to record its activities."""
from autogpt.core.logging.base import Logger

from autogpt.core.status import Status

status = Status.INTERFACE_DONE
handover_notes = "Interface has been created. Basic example needs to be created."