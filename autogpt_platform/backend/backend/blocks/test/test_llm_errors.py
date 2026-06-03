import httpx
import openai
import pytest

import backend.blocks.llm as llm
from backend.blocks.llm_errors import (
    format_llm_error_message,
    is_invalid_model_error,
)


def _make_openai_status_error(message: str, code: str | None = None, status: int = 400):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.openai.com/v1/chat")
    )
    body = {"code": code} if code else None
    return openai.APIStatusError(message, response=response, body=body)


@pytest.mark.parametrize(
    "message",
    [
        "400 invalid model ID: claude-haiku-4-5-20251001",
        "Unknown model: gpt-imaginary",
        "The model `gpt-4o-mini-old` does not exist",
        "model gpt-4o-old not found",
        "This model is no longer available",
        "deprecated model: text-davinci-003",
        "unsupported model requested",
        "model gpt-4o-old is not available",
    ],
)
def test_is_invalid_model_error_matches_known_patterns(message: str):
    assert is_invalid_model_error(ValueError(message)) is True


def test_is_invalid_model_error_matches_structured_code():
    error = _make_openai_status_error(
        "Error code: 404 - resource missing", code="model_not_found"
    )

    assert is_invalid_model_error(error) is True


@pytest.mark.parametrize(
    "message",
    [
        "400 prompt rejected by moderation",
        "The model rate limit does not exist for this plan",
        "The dataset does not exist",
        "User profile no longer available",
        "model latency is high, please retry",
    ],
)
def test_is_invalid_model_error_ignores_non_model_errors(message: str):
    assert is_invalid_model_error(ValueError(message)) is False


def test_format_llm_error_message_appends_guidance_for_invalid_model():
    error = ValueError("invalid model ID: claude-haiku-4-5-20251001")

    message = format_llm_error_message(error, llm.LlmModel.CLAUDE_4_5_HAIKU)

    assert "invalid or no longer available" in message
    assert "Check Anthropic's current model list" in message
    assert "update the block's model configuration" in message


def test_format_llm_error_message_interpolates_openai_provider():
    error = ValueError("invalid model ID: gpt-imaginary")

    message = format_llm_error_message(error, llm.LlmModel.GPT4O)

    assert "Check OpenAI's current model list" in message


def test_format_llm_error_message_uses_precomputed_flag():
    error = ValueError("prompt rejected by moderation")

    message = format_llm_error_message(
        error, llm.LlmModel.CLAUDE_4_5_HAIKU, is_invalid_model=True
    )

    assert "Check Anthropic's current model list" in message


def test_format_llm_error_message_preserves_generic_errors():
    error = ValueError("prompt rejected by moderation")

    message = format_llm_error_message(error, llm.LlmModel.CLAUDE_4_5_HAIKU)

    assert message == "Error calling LLM: prompt rejected by moderation"
