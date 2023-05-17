import enum
from typing import Any

from pydantic import BaseModel

from autogpt.core.resource.model_providers.schema import (
    LanguageModelMessage,
    LanguageModelProviderModelResponse,
)


class LanguageModelClassification(str, enum.Enum):
    """The LanguageModelClassification is a functional description of the model.

    This is used to determine what kind of model to use for a given prompt.
    Sometimes we prefer a faster or cheaper model to accomplish a task when
    possible.

    """

    FAST_MODEL: str = "fast_model"
    SMART_MODEL: str = "smart_model"


class LanguageModelPrompt(BaseModel):
    messages: list[LanguageModelMessage]
    tokens_used: int


class LanguageModelResponse(LanguageModelProviderModelResponse):
    """Standard response struct for a response from a language model."""


class PlanningContext(BaseModel):
    progress: Any  # To be defined (maybe here, as this might be a good place for summarization)
    last_command_result: Any  # To be defined in the command interface
    memories: Any  # List[Memory] # To be defined in the memory interface
    user_feedback: Any  # Probably just a raw string


class ReflectionContext(BaseModel):
    # Using existing args here
    reasoning: str
    plan: list[str]
    thoughts: str
    criticism: str
