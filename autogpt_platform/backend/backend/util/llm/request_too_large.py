"""An LLM request bigger than the provider accepts in one call.

OpenAI answers a request larger than the account's tokens-per-minute limit
with a 429 that shares its code (``rate_limit_exceeded``) and type with an
ordinary rate limit, so only its message tells the two apart. Anthropic and
Groq answer an oversized request with a 413. The same request is refused every
time, so it is not retried and the user is told to make it smaller.
"""

import re

import anthropic
import groq
import openai

# OpenAI's and Groq's wording for one request over a per-minute token limit.
_TOO_LARGE_PHRASE = "request too large"

# The sizes the provider reported, e.g. "Limit 400000, Requested 520131".
_REPORTED_SIZES_RE = re.compile(r"limit (\d+), requested (\d+)", re.IGNORECASE)


class RequestTooLargeError(ValueError):
    """The provider refused the request for its size; retrying cannot help.

    ``str()`` is the user-facing sentence. It never quotes the provider's
    text, which names the platform's own account; that error is on
    ``__cause__``.
    """

    def __init__(self, provider_name: str, provider_message: str = "") -> None:
        sizes = _REPORTED_SIZES_RE.search(provider_message)
        reported = (
            f" It was {int(sizes.group(2)):,} tokens; the limit is "
            f"{int(sizes.group(1)):,}."
            if sizes
            else ""
        )
        super().__init__(
            f"This request is larger than {provider_name} accepts in one call, "
            f"so trying again won't help.{reported} Shorten the input (less "
            "text, or fewer or smaller files), or choose a model with a higher "
            "limit."
        )


def is_request_too_large(error: BaseException) -> bool:
    """True when the provider refused the request for its size."""
    if not isinstance(
        error, (openai.APIStatusError, anthropic.APIStatusError, groq.APIStatusError)
    ):
        return False
    if error.status_code == 413:
        return True
    return error.status_code == 429 and is_request_too_large_text(str(error))


def is_request_too_large_text(text: str) -> bool:
    """The same check for a failure that only reaches us as text."""
    return _TOO_LARGE_PHRASE in text.lower()
