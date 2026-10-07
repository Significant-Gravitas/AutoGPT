"""A platform-owned LLM provider account running out of money.

When the OpenRouter / OpenAI / Anthropic account the platform pays for runs
dry, the provider answers every request with a billing refusal ("This request
requires more credits ... visit openrouter.ai/settings/credits"). Passed
through verbatim, that tells a user to go and buy credits on a site they have
no account with, for a bill that is ours. It is an outage on our side, so the
user gets one plain sentence and we get paged.

This is deliberately separate from the platform's own credit checks
(``InsufficientBalanceError``, the copilot usage limit). Those are the user's
balance and must keep telling the user so; nothing here matches their text.
"""

import logging
import re

import anthropic
import openai
import sentry_sdk

logger = logging.getLogger(__name__)

PROVIDER_UNAVAILABLE_MESSAGE = (
    "The AI model provider is temporarily unavailable. We've been alerted and "
    "are working on it. Please try again shortly."
)

# Stream error code the copilot sends alongside the message above.
PROVIDER_UNAVAILABLE_CODE = "provider_unavailable"

# Stable Sentry title -- never interpolate into it, the ids go in json_fields.
PROVIDER_OUT_OF_CREDITS_ALERT = "LLM provider account is out of credits"

# Provider codes that mean "the account has no money", as opposed to a rate
# limit that lifts on its own (OpenAI sends both as HTTP 429).
_BILLING_ERROR_CODES = frozenset(
    {"insufficient_quota", "billing_error", "credit_balance_exhausted"}
)

# Anthropic's one refusal with no billing code: a 400 invalid_request_error.
# Only read from the message field of an Anthropic error body.
_ANTHROPIC_LOW_BALANCE = "credit balance is too low"

# For failures that only reach us as text: the Claude CLI, which talks to
# Anthropic or OpenRouter, so only their wording is here. OpenAI's "exceeded
# your current quota" is not: Gemini uses it for an ordinary 429 rate limit,
# and OpenAI's real case arrives typed with code insufficient_quota.
_TEXT_ONLY_PATTERNS = (
    _ANTHROPIC_LOW_BALANCE,
    # OpenRouter
    "requires more credits",
    "openrouter.ai/settings/credits",
)

# A bare 402 as the Claude CLI and httpx render it.
_BARE_402_RE = re.compile(
    r"(?:error code:|api error:|status code)\s*402\b|\b402 payment required\b"
)


def is_provider_out_of_credits(error: BaseException | str | None) -> bool:
    """True when a provider refused because the account it bills is empty.

    An exception counts only through a typed SDK error in its chain, judged on
    its structured fields alone: any error's text can quote an upstream
    (OpenRouter relays Gemini's rate limit verbatim in ``metadata.raw``), the
    user's own input or a fetched page. A ``str`` is for the Claude CLI's own
    error result, which only reaches us as text.
    """
    if error is None:
        return False
    if isinstance(error, str):
        return _text_matches(error)
    typed = _typed_provider_error(error)
    return typed is not None and _typed_error_is_billing(typed)


def _typed_provider_error(
    error: BaseException,
) -> openai.APIError | anthropic.APIError | None:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, (openai.APIError, anthropic.APIError)):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _typed_error_is_billing(error: openai.APIError | anthropic.APIError) -> bool:
    if isinstance(error, (openai.APIStatusError, anthropic.APIStatusError)):
        if error.status_code == 402:
            return True
    # A mid-stream OpenAI SDK error has no status, only the error object.
    if isinstance(error, openai.APIError) and error.code in (402, "402"):
        return True
    nested = _error_object(error.body)
    if {nested.get("code"), nested.get("type")} & _BILLING_ERROR_CODES:
        return True
    if isinstance(error, anthropic.APIError):
        message = nested.get("message")
        return isinstance(message, str) and _ANTHROPIC_LOW_BALANCE in message.lower()
    return False


def _error_object(body: object) -> dict:
    if not isinstance(body, dict):
        return {}
    nested = body.get("error", body)
    return nested if isinstance(nested, dict) else {}


def _text_matches(text: str) -> bool:
    lower = text.lower()
    return bool(_BARE_402_RE.search(lower)) or any(
        pattern in lower for pattern in _TEXT_ONLY_PATTERNS
    )


def report_provider_out_of_credits(
    *,
    provider: str,
    model: str | None,
    surface: str,
    error: BaseException | str,
    **context: object,
) -> None:
    """Log the refusal at ERROR under its own Sentry fingerprint.

    One issue per provider, so an empty OpenRouter account is a single loud
    alert rather than one event per user per model folded into whatever
    generic "LLM call failed" issue it would otherwise land in.
    """
    fields = {
        "provider": provider,
        "model": model,
        "surface": surface,
        "error": str(error) or repr(error),
        **context,
    }
    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["llm-provider-out-of-credits", provider]
        scope.set_tag("llm_provider", provider)
        scope.set_tag("llm_surface", surface)
        if model:
            scope.set_tag("llm_model", model)
        logger.error(
            PROVIDER_OUT_OF_CREDITS_ALERT, extra={"json_fields": fields}, stacklevel=2
        )


class ProviderUnavailableError(RuntimeError):
    """The provider refused for billing reasons on a platform-owned key.

    ``str()`` is the user-facing sentence; the provider's own text is only on
    ``__cause__`` and in the alert.
    """

    def __init__(self, message: str = PROVIDER_UNAVAILABLE_MESSAGE) -> None:
        super().__init__(message)
