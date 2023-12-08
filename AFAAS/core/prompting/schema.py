import enum

# from pydantic import BaseModel, Field

# from AFAAS.app.core.tools.schema import ToolResult
# from AFAAS.app.core.resource.model_providers.chat_schema import (
#     ChatMessage,
#     ChatMessageDict,
#     ChatModelResponse,
#     CompletionModelFunction,
# )


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
