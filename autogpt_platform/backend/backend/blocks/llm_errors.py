import re
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from backend.blocks.llm import LlmModel

# OpenAI exposes a structured error code that reliably flags an invalid or
# retired model ID, independent of the (translatable, changeable) message text.
INVALID_MODEL_ERROR_CODES = frozenset(
    {"model_not_found", "model_not_supported", "unsupported_model"}
)

# Message fallback for providers without structured codes. The model-ID branch
# allows at most one token (the offending model ID, often backtick/quote
# wrapped) between "model" and the failure phrase, so multi-word phrases such as
# "model rate limit does not exist" do not false-positive.
_INVALID_MODEL_MESSAGE_RE = re.compile(
    r"invalid model"
    r"|unknown model"
    r"|unsupported model"
    r"|deprecated model"
    r"|model\b(?:\s+[`'\"]?[\w./:-]+[`'\"]?)?\s+"
    r"(?:not found|does not exist|no longer available|"
    r"is not available|not supported)",
    re.IGNORECASE,
)


def is_invalid_model_error(error: Exception) -> bool:
    """Return whether a provider error looks like an invalid or retired model ID."""
    if (
        isinstance(error, openai.APIStatusError)
        and error.code in INVALID_MODEL_ERROR_CODES
    ):
        return True
    return bool(_INVALID_MODEL_MESSAGE_RE.search(str(error)))


def format_llm_error_message(
    error: Exception,
    llm_model: "LlmModel",
    is_invalid_model: bool | None = None,
) -> str:
    """Append actionable guidance to model-ID validation errors.

    Pass ``is_invalid_model`` when the caller has already classified the error
    to avoid re-running the detection.
    """
    message = f"Error calling LLM: {error}"
    if is_invalid_model is None:
        is_invalid_model = is_invalid_model_error(error)
    if not is_invalid_model:
        return message

    provider_name = llm_model.metadata.provider_name
    return (
        f"{message}\n\n"
        f"The configured model ID `{llm_model.value}` appears to be invalid "
        f"or no longer available. Check {provider_name}'s current model list and "
        "update the block's model configuration."
    )
