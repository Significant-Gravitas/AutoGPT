"""An LLM provider refusing a prompt for its content.

OpenAI's policy check and OpenRouter's moderation reject a prompt they judge
against their usage policy. That is the user's prompt to reword, not a fault
of ours, so it reaches them as one plain sentence and stays out of Sentry.
"""

import openai

PROMPT_REFUSED_MESSAGE = (
    "The AI model's provider refused this prompt as possibly violating its "
    "usage policy. Rewording the prompt may help."
)

# OpenAI's error codes for a prompt its policy check rejected.
_OPENAI_REFUSAL_CODES = frozenset({"invalid_prompt", "content_policy_violation"})


class PromptRefusedError(ValueError):
    """The provider refused the prompt's content.

    ``str()`` is the user-facing sentence, which never quotes the prompt; the
    provider's own error is on ``__cause__``.
    """

    def __init__(self, message: str = PROMPT_REFUSED_MESSAGE) -> None:
        super().__init__(message)


def is_prompt_refused(error: BaseException) -> bool:
    """True when the provider refused the prompt for its content.

    Judged on the SDK error's structured fields only, never its text, which
    can quote the prompt itself.
    """
    if not isinstance(error, openai.APIStatusError):
        return False
    if error.code in _OPENAI_REFUSAL_CODES:
        return True
    return error.status_code == 403 and _is_openrouter_moderation(error.body)


def _is_openrouter_moderation(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    metadata = body.get("metadata")
    return isinstance(metadata, dict) and "flagged_input" in metadata
