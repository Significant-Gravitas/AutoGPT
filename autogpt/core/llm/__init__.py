"""The language model acts as the core intelligence of the Agent."""
from autogpt.core.llm.base import LanguageModel
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Next up is creating a basic implementation."
    ),
)
