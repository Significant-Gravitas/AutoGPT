"""The language model acts as the core intelligence of the Agent."""
from autogpt.core.llm.base import LanguageModel

import autogpt.core.status

status = autogpt.core.status.Status.INTERFACE_DONE
handover_notes = "Interface has been created. Next up is creating a basic implementation."
