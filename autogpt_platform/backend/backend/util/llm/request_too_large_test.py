import anthropic
import groq
import httpx
import openai
import pytest

from backend.util.llm.request_too_large import (
    RequestTooLargeError,
    is_request_too_large,
)

_REQUEST = httpx.Request("POST", "https://provider.test")

_OPENAI_TOO_LARGE = (
    "Request too large for gpt-4.1-mini-2025-04-14 in organization org-test on "
    "tokens per min (TPM): Limit 400000, Requested 520131."
)
_OPENAI_RATE_LIMITED = (
    "Rate limit reached for gpt-4.1-mini-2025-04-14 in organization org-test on "
    "tokens per min (TPM): Limit 400000, Used 390000, Requested 20000."
)
_GROQ_TOO_LARGE = (
    "Request too large for model `llama-3.3-70b-versatile` on tokens per minute "
    "(TPM): Limit 6000, Requested 9000, please reduce your message size."
)


def _openai(status: int, message: str) -> openai.APIStatusError:
    return openai.APIStatusError(
        message,
        response=httpx.Response(status, request=_REQUEST),
        body={"message": message, "code": "rate_limit_exceeded"},
    )


def _anthropic(status: int, message: str) -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        message, response=httpx.Response(status, request=_REQUEST), body=None
    )


def _groq(status: int, message: str) -> groq.APIStatusError:
    return groq.APIStatusError(
        message, response=httpx.Response(status, request=_REQUEST), body=None
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        pytest.param(_openai(429, _OPENAI_TOO_LARGE), True, id="openai-tpm"),
        pytest.param(_openai(429, _OPENAI_RATE_LIMITED), False, id="openai-rate"),
        pytest.param(_openai(400, _OPENAI_TOO_LARGE), False, id="openai-400"),
        pytest.param(_anthropic(413, "too large"), True, id="anthropic-413"),
        pytest.param(_anthropic(429, "rate limited"), False, id="anthropic-429"),
        pytest.param(_groq(413, _GROQ_TOO_LARGE), True, id="groq-413"),
        pytest.param(_groq(429, "Rate limit reached"), False, id="groq-429"),
        pytest.param(RuntimeError(_OPENAI_TOO_LARGE), False, id="untyped"),
    ],
)
def test_is_request_too_large(error: BaseException, expected: bool):
    assert is_request_too_large(error) is expected


def test_message_names_the_reported_sizes_but_not_the_account():
    message = str(RequestTooLargeError("Groq", _GROQ_TOO_LARGE))

    assert message.startswith("This request is larger than Groq accepts")
    assert "It was 9,000 tokens; the limit is 6,000." in message
    assert "llama-3.3-70b-versatile" not in message


def test_message_without_reported_sizes():
    message = str(RequestTooLargeError("Anthropic"))

    assert "tokens" not in message
    assert "Shorten the input" in message
