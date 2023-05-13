"""The language model acts as the core intelligence of the Agent."""
from autogpt.core.model.base import ModelInfo, ModelResponse, ModelType
from autogpt.core.model.embedding import (
    EmbeddingModel,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
)
from autogpt.core.model.language import (
    LanguageModel,
    LanguageModelInfo,
    LanguageModelResponse,
)
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Next up is creating a basic implementation.\n"
        "5/11: Refactored interface to split out embedding and language models.\n"
        "      Created a basic implementation of the language model and work out several utilities.\n"
        "5/12: Restructure the relationship between the language/embedding models and providers and figure\n"
        "      out how to do credentials in a reasonable way.\n"
    ),
)
