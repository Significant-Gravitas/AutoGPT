import enum

from pydantic import BaseModel, Field

from autogpt.core.ability.schema import AbilityResult
from autogpt.core.resource.model_providers.schema import (
    ChatMessage,
    ChatMessageDict,
    ChatModelResponse,
    CompletionModelFunction,
)


class LanguageModelClassification(str, enum.Enum):
    """The LanguageModelClassification is a functional description of the model.

    This is used to determine what kind of model to use for a given prompt.
    Sometimes we prefer a faster or cheaper model to accomplish a task when
    possible.
    """

    FAST_MODEL_4K: str = "FAST_MODEL_4K"
    FAST_MODEL_16K: str = "FAST_MODEL_16K"
    FAST_MODEL_FINE_TUNED_4K: str = "FAST_MODEL_FINE_TUNED_4K"
    SMART_MODEL_8K: str = "SMART_MODEL_8K"
    SMART_MODEL_32K: str = "SMART_MODEL_32K"


class ChatPrompt(BaseModel):
    messages: list[ChatMessage]
    functions: list[CompletionModelFunction] = Field(default_factory=list)
    function_call: str
    default_function_call: str

    def raw(self) -> list[ChatMessageDict]:
        return [m.dict() for m in self.messages]

    def __str__(self):
        return "\n\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in self.messages
        )
