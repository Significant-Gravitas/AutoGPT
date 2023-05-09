"""The command system provides a way to extend the functionality of the AI agent."""
from autogpt.core.command.base import Command, CommandRegistry
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.IN_PROGRESS,
    handoff_notes="More work is needed, basic ideas are in place.",
)
