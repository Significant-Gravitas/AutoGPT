"""Tests for the Avian LLM provider integration.

Avian uses an OpenAI-compatible API (base URL https://api.avian.io/v1) and is
handled through the shared AIBlockBase hierarchy in blocks/llm.py, the same way
OpenAI, Anthropic, Groq, etc. are handled.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import SecretStr

import backend.blocks.llm as llm
from backend.data.model import APIKeyCredentials

AVIAN_CREDENTIALS = APIKeyCredentials(
    id="test-avian-id",
    provider="avian",
    api_key=SecretStr("mock-avian-api-key"),
    title="Mock Avian API key",
    expires_at=None,
)


def _mock_openai_chat_response(
    content: str = "Hello from Avian",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
):
    """Build a fake openai.ChatCompletion-like response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestAvianModelMetadata:
    """Validate model metadata for all Avian models."""

    @pytest.mark.parametrize(
        "model",
        [
            llm.LLMModel.AVIAN_DEEPSEEK_V3_2,
            llm.LLMModel.AVIAN_KIMI_K2_5,
            llm.LLMModel.AVIAN_GLM_5,
            llm.LLMModel.AVIAN_MINIMAX_M2_5,
        ],
    )
    def test_avian_model_has_metadata(self, model):
        """Every Avian model must have a corresponding entry in MODEL_METADATA."""
        assert model in llm.MODEL_METADATA

    @pytest.mark.parametrize(
        "model",
        [
            llm.LLMModel.AVIAN_DEEPSEEK_V3_2,
            llm.LLMModel.AVIAN_KIMI_K2_5,
            llm.LLMModel.AVIAN_GLM_5,
            llm.LLMModel.AVIAN_MINIMAX_M2_5,
        ],
    )
    def test_avian_model_provider_is_avian(self, model):
        """All Avian models must declare provider='avian'."""
        metadata = llm.MODEL_METADATA[model]
        assert metadata.provider == "avian"

    def test_avian_in_llm_provider_name(self):
        """ProviderName.AVIAN must be listed in LLMProviderName."""
        from backend.integrations.providers import ProviderName

        assert ProviderName.AVIAN in llm.LLMProviderName.__args__
