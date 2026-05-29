from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.blocks.llm import LlmModel

INVALID_MODEL_ERROR_PATTERNS = (
    "invalid model",
    "invalid model id",
    "invalid model_id",
    "unknown model",
    "model not found",
    "does not exist",
    "no longer available",
    "deprecated model",
    "unsupported model",
)


def is_invalid_model_error(error: Exception) -> bool:
    """Return whether a provider error looks like an invalid or retired model ID."""
    message = str(error).lower()
    return "model" in message and any(
        pattern in message for pattern in INVALID_MODEL_ERROR_PATTERNS
    )


def format_llm_error_message(error: Exception, llm_model: "LlmModel") -> str:
    """Append actionable guidance to model-ID validation errors."""
    message = f"Error calling LLM: {error}"
    if not is_invalid_model_error(error):
        return message

    provider_name = llm_model.metadata.provider_name
    return (
        f"{message} The configured model ID `{llm_model.value}` appears to be invalid "
        f"or no longer available. Check {provider_name}'s current model list and "
        "update the block's model configuration."
    )
