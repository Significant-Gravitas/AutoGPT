"""The language model acts as the core intelligence of the Agent."""
from autogpt.core.llm.base import LanguageModel

from autogpt.core.status import Status

status = Status.INTERFACE_DONE
handover_notes = "Interface has been created. Next up is creating a basic implementation."
