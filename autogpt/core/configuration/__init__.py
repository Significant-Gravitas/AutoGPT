"""The configuration encapsulates settings for all Agent subsystems."""
from autogpt.core.configuration.base import Configuration
import autogpt.core.status

status = autogpt.core.status.Status.INTERFACE_DONE
handover_notes = "Interface has been created. Basic example needs to be created."