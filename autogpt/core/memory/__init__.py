"""The memory subsystem manages the Agent's long-term memory."""
from autogpt.core.memory.base import MemoryBackend
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.TODO,
    handoff_notes=(
        "Before times: The memory subsystem has not been started yet. There is ongoing research work\n"
        "              on this part of the system and we're holding interface design until that work\n"
        "              is at least partially resolved.\n"
    ),
)
