"""The language model acts as the core intelligence of the Agent."""
from autogpt.core.embedding.base import EmbeddingModel, EmbeddingModelResponse
from autogpt.core.embedding.simple import EmbeddingModelSettings, SimpleEmbeddingModel
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
        "5/14: Use pydantic models for configuration.\n"
        "5/16: Language model works for the objective prompt. There's an unresolved token count relationship\n"
        "      between the language model and the planner class. Need to get model info to the planner so it \n"
        "      can make decisions about how to use the prompt space it has available. Method TBD.\n"
    ),
)
