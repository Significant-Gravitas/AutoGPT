"""Tests for the Avian LLM provider integration.

Avian uses an OpenAI-compatible API (base URL https://api.avian.io/v1) and is
handled through the shared AIBlockBase hierarchy in blocks/llm.py, the same way
OpenAI, Anthropic, Groq, etc. are handled.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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


class TestAvianLlmCall:
    """Test the Avian branch inside llm_call."""

    @pytest.mark.asyncio
    async def test_basic_completion(self):
        """Avian provider returns a valid LLMResponse via the OpenAI SDK."""
        mock_response = _mock_openai_chat_response()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await llm.llm_call(
                credentials=AVIAN_CREDENTIALS,
                llm_model=llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
                prompt=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert isinstance(result, llm.LLMResponse)
        assert result.response == "Hello from Avian"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_uses_correct_base_url(self):
        """The OpenAI client must be created with base_url=https://api.avian.io/v1."""
        mock_response = _mock_openai_chat_response()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            await llm.llm_call(
                credentials=AVIAN_CREDENTIALS,
                llm_model=llm.LlmModel.AVIAN_KIMI_K2_5,
                prompt=[{"role": "user", "content": "Hello"}],
                max_tokens=50,
            )

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://api.avian.io/v1"

    @pytest.mark.asyncio
    async def test_glm5_model_name_remapped(self):
        """AVIAN_GLM_5 enum value 'avian/glm-5' must be sent as 'z-ai/glm-5'."""
        mock_response = _mock_openai_chat_response()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_create = AsyncMock(return_value=mock_response)
            mock_client.chat.completions.create = mock_create

            await llm.llm_call(
                credentials=AVIAN_CREDENTIALS,
                llm_model=llm.LlmModel.AVIAN_GLM_5,
                prompt=[{"role": "user", "content": "Test"}],
                max_tokens=100,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "z-ai/glm-5"

    @pytest.mark.asyncio
    async def test_non_remapped_model_uses_enum_value(self):
        """Models without a remap entry use the raw enum value as the API model."""
        mock_response = _mock_openai_chat_response()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_create = AsyncMock(return_value=mock_response)
            mock_client.chat.completions.create = mock_create

            await llm.llm_call(
                credentials=AVIAN_CREDENTIALS,
                llm_model=llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
                prompt=[{"role": "user", "content": "Test"}],
                max_tokens=100,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek/deepseek-v3.2"

    @pytest.mark.asyncio
    async def test_empty_choices_raises(self):
        """An empty choices list from the Avian API should raise ValueError."""
        mock_response = MagicMock()
        mock_response.choices = []

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            with pytest.raises(ValueError, match="Avian API error"):
                await llm.llm_call(
                    credentials=AVIAN_CREDENTIALS,
                    llm_model=llm.LlmModel.AVIAN_MINIMAX_M2_5,
                    prompt=[{"role": "user", "content": "Hello"}],
                    max_tokens=100,
                )

    @pytest.mark.asyncio
    async def test_json_output_mode(self):
        """force_json_output sends response_format={'type':'json_object'}."""
        mock_response = _mock_openai_chat_response(content='{"key": "value"}')

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_create = AsyncMock(return_value=mock_response)
            mock_client.chat.completions.create = mock_create

            await llm.llm_call(
                credentials=AVIAN_CREDENTIALS,
                llm_model=llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
                prompt=[{"role": "user", "content": "Return JSON"}],
                max_tokens=100,
                force_json_output=True,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_no_json_output_mode(self):
        """Without force_json_output the response_format should be None."""
        mock_response = _mock_openai_chat_response()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_create = AsyncMock(return_value=mock_response)
            mock_client.chat.completions.create = mock_create

            await llm.llm_call(
                credentials=AVIAN_CREDENTIALS,
                llm_model=llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
                prompt=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] is None


class TestAvianModelMetadata:
    """Validate model metadata for all Avian models."""

    @pytest.mark.parametrize(
        "model",
        [
            llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
            llm.LlmModel.AVIAN_KIMI_K2_5,
            llm.LlmModel.AVIAN_GLM_5,
            llm.LlmModel.AVIAN_MINIMAX_M2_5,
        ],
    )
    def test_avian_model_has_metadata(self, model):
        """Every Avian model must have a corresponding entry in MODEL_METADATA."""
        assert model in llm.MODEL_METADATA

    @pytest.mark.parametrize(
        "model",
        [
            llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
            llm.LlmModel.AVIAN_KIMI_K2_5,
            llm.LlmModel.AVIAN_GLM_5,
            llm.LlmModel.AVIAN_MINIMAX_M2_5,
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


class TestAvianBlockIntegration:
    """Test that existing AI blocks work end-to-end with Avian credentials."""

    @pytest.mark.asyncio
    async def test_structured_response_block_with_avian(self):
        """AIStructuredResponseGeneratorBlock runs correctly with Avian provider."""
        block = llm.AIStructuredResponseGeneratorBlock()

        async def mock_llm_call(*args, **kwargs):
            return llm.LLMResponse(
                raw_response="",
                prompt=[],
                response='<json_output id="test123456">{"name": "Avian", "status": "ok"}</json_output>',
                tool_calls=None,
                prompt_tokens=12,
                completion_tokens=8,
                reasoning=None,
            )

        block.llm_call = mock_llm_call  # type: ignore

        avian_creds_input = {
            "provider": AVIAN_CREDENTIALS.provider,
            "id": AVIAN_CREDENTIALS.id,
            "type": AVIAN_CREDENTIALS.type,
        }

        input_data = llm.AIStructuredResponseGeneratorBlock.Input(
            prompt="Describe yourself",
            expected_format={"name": "provider name", "status": "status"},
            model=llm.LlmModel.AVIAN_DEEPSEEK_V3_2,
            credentials=avian_creds_input,  # type: ignore
        )

        outputs = {}
        with patch("secrets.token_hex", return_value="test123456"):
            async for name, data in block.run(
                input_data, credentials=AVIAN_CREDENTIALS
            ):
                outputs[name] = data

        assert outputs["response"] == {"name": "Avian", "status": "ok"}
        assert block.execution_stats.input_token_count == 12
        assert block.execution_stats.output_token_count == 8

    @pytest.mark.asyncio
    async def test_text_generator_block_with_avian(self):
        """AITextGeneratorBlock runs correctly with Avian provider."""
        from backend.data.model import NodeExecutionStats

        block = llm.AITextGeneratorBlock()

        async def mock_llm_call(input_data, credentials):
            block.execution_stats = NodeExecutionStats(
                input_token_count=20,
                output_token_count=30,
                llm_call_count=1,
            )
            return "Generated text from Avian"

        block.llm_call = mock_llm_call  # type: ignore

        avian_creds_input = {
            "provider": AVIAN_CREDENTIALS.provider,
            "id": AVIAN_CREDENTIALS.id,
            "type": AVIAN_CREDENTIALS.type,
        }

        input_data = llm.AITextGeneratorBlock.Input(
            prompt="Write something",
            model=llm.LlmModel.AVIAN_KIMI_K2_5,
            credentials=avian_creds_input,  # type: ignore
        )

        outputs = {}
        async for name, data in block.run(input_data, credentials=AVIAN_CREDENTIALS):
            outputs[name] = data

        assert outputs["response"] == "Generated text from Avian"
        assert block.execution_stats.input_token_count == 20
        assert block.execution_stats.output_token_count == 30
