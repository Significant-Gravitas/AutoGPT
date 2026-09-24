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

# Matched case-insensitively against error text, for failures that only reach
# us as text (the Claude CLI) or whose status was lost in a wrapper. Each one
# is provider wording; none can appear in the platform's own balance errors.
_BILLING_ERROR_PATTERNS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_error",
    # OpenAI
    "exceeded your current quota",
    # Anthropic
    "credit balance is too low",
    # OpenRouter
    "requires more credits",
    "can only afford",
    "openrouter.ai/settings/credits",
    # A bare 402 as the OpenAI SDK, the Claude CLI and httpx render it
    "error code: 402",
    "api error: 402",
    "status code 402",
    "402 payment required",
)


def is_provider_out_of_credits(error: BaseException | str | None) -> bool:
    """True when a provider refused because the account it bills is empty."""
    if error is None:
        return False
    if isinstance(error, (openai.APIStatusError, anthropic.APIStatusError)):
        if error.status_code == 402 or _body_names_billing(error.body):
            return True
    if isinstance(error, openai.APIError) and error.code in _BILLING_ERROR_CODES:
        return True
    return _text_matches(str(error))


def _body_names_billing(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    nested = body.get("error", body)
    return isinstance(nested, dict) and bool(
        {nested.get("code"), nested.get("type")} & _BILLING_ERROR_CODES
    )


def _text_matches(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in _BILLING_ERROR_PATTERNS)


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
