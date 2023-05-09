"""The memory subsystem manages the Agent's long-term memory."""
from autogpt.core.memory.base import MemoryBackend
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.TODO,
    handoff_notes="The memory subsystem has not been started yet.",
)
