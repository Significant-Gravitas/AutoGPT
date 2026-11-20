import re
from typing import TYPE_CHECKING

import anthropic
import openai

if TYPE_CHECKING:
    from backend.data.llm_registry.llm_models import LLMModel

# OpenAI exposes a structured error code that reliably flags an invalid or
# retired model ID, independent of the (translatable, changeable) message text.
INVALID_MODEL_ERROR_CODES = frozenset(
    {"model_not_found", "model_not_supported", "unsupported_model"}
)

# Other statuses retain their existing retry and user-error handling.
INVALID_MODEL_STATUS_CODES = frozenset({400, 404})

# Bare model IDs need a separator or digit; quoted IDs are unambiguous.
# This excludes nouns such as "model config" and "model response".
_INVALID_MODEL_MESSAGE_RE = re.compile(
    r"\b(?:invalid|unknown|unsupported|deprecated|retired|no such) model"
    r"(?:[ _]id)?(?:\s*[:=]|\s*$|\s+requested\b)"
    r"|\bis not a valid model\b"
    r"|\bmodel(?:[ _]id)?(?:\s+(?:[`'\"][^`'\"\r\n]+[`'\"]|"
    r"[\w./:-]*[-./:\d][\w./:-]*))?\s+"
    r"(?:not found|does not exist|(?:is )?no longer available|"
    r"is not available|not supported|"
    r"(?:has been |was |is )?(?:deprecated|retired))",
    re.IGNORECASE,
)

# Context-window overflows mention the model too ("This model's maximum context
# length is 8192 tokens..."). They are retryable with reduced max_tokens and
# must never classify as invalid-model errors.
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length|context length|token limit", re.IGNORECASE
)

_MODEL_LIST_URLS = {
    "openai": "https://developers.openai.com/api/docs/models",
    "anthropic": "https://platform.claude.com/docs/en/about-claude/models/overview",
}


def is_invalid_model_error(error: Exception) -> bool:
    """Return whether a provider error looks like an invalid or retired model ID."""
    if not isinstance(error, (anthropic.APIStatusError, openai.APIStatusError)):
        return False
    if error.status_code not in INVALID_MODEL_STATUS_CODES:
        return False
    body = error.body
    if isinstance(body, dict):
        body = body.get("error", body)
        if not isinstance(body, dict):
            return False
        code = body.get("code")
        message = body.get("message", "")
    elif body is None or isinstance(body, str):
        code = None
        message = error.message
    else:
        return False
    if not isinstance(message, str):
        return False
    if _CONTEXT_LENGTH_RE.search(message):
        return False
    if isinstance(code, str) and code in INVALID_MODEL_ERROR_CODES:
        return True
    if isinstance(error, anthropic.NotFoundError) and re.fullmatch(
        r"model:\s*\S+", message, re.IGNORECASE
    ):
        return True
    return bool(_INVALID_MODEL_MESSAGE_RE.search(message))


def format_llm_error_message(
    error: Exception,
    llm_model: "LLMModel",
    is_invalid_model: bool | None = None,
) -> str:
    """Replace model-ID errors with actionable guidance; preserve other errors.

    Pass ``is_invalid_model`` when the caller has already classified the error
    to avoid re-running the detection.
    """
    if is_invalid_model is None:
        is_invalid_model = is_invalid_model_error(error)
    if not is_invalid_model:
        return f"Error calling LLM: {error}"

    provider_name = llm_model.metadata.provider_name
    model_list_url = _MODEL_LIST_URLS.get(llm_model.metadata.provider)
    model_list = f" ({model_list_url})" if model_list_url else ""
    return (
        f"Error calling LLM: The configured model ID `{llm_model.value}` appears "
        "to be invalid or no longer available to this API key. "
        f"Check {provider_name}'s current model list{model_list} and "
        "update the block's model configuration."
    )
