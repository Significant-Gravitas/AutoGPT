import backend.blocks.llm as llm
from backend.blocks.llm_errors import (
    format_llm_error_message,
    is_invalid_model_error,
)


def test_is_invalid_model_error_matches_known_patterns():
    error = ValueError("400 invalid model ID: claude-haiku-4-5-20251001")

    assert is_invalid_model_error(error) is True


def test_is_invalid_model_error_ignores_non_model_errors():
    error = ValueError("400 prompt rejected by moderation")

    assert is_invalid_model_error(error) is False


def test_format_llm_error_message_appends_guidance_for_invalid_model():
    error = ValueError("invalid model ID: claude-haiku-4-5-20251001")

    message = format_llm_error_message(error, llm.LlmModel.CLAUDE_4_5_HAIKU)

    assert "invalid or no longer available" in message
    assert "Check Anthropic's current model list" in message
    assert "update the block's model configuration" in message


def test_format_llm_error_message_preserves_generic_errors():
    error = ValueError("prompt rejected by moderation")

    message = format_llm_error_message(error, llm.LlmModel.CLAUDE_4_5_HAIKU)

    assert message == "Error calling LLM: prompt rejected by moderation"
