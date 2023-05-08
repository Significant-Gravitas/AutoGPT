"""The command system provides a way to extend the functionality of the AI agent."""
from autogpt.core.command.base import Command, CommandRegistry
from autogpt.core.status import Status

status = Status.IN_PROGRESS
handover_notes = "More work is needed, basic ideas are in place."
