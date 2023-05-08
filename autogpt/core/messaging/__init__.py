"""The messaging system provides a protocol for Agent communication with other agents and users."""
from autogpt.core.messaging.base import Message, MessageBroker

from autogpt.core.status import Status

status = Status.BASIC_DONE
handover_notes = "Interface has been completed and a basic implementation has been created."
