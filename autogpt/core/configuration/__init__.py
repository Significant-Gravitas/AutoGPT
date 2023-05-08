"""The configuration encapsulates settings for all Agent subsystems."""
from autogpt.core.configuration.base import Configuration
from autogpt.core.status import Status

status = Status.INTERFACE_DONE
handover_notes = "Interface has been created. Basic example needs to be created."