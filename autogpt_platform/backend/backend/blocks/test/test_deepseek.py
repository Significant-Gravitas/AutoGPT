from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.deepseek import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DeepSeekBlock,
    DeepSeekModel,
)
from backend.data.model import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError


def _make_input(**overrides) -> dict:
    defaults = {
        "prompt": "Hello DeepSeek!",
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    defaults.update(overrides)
    return defaults


class TestDeepSeekModelFallback:
    """Tests for model fallback and input validation."""

    def test_invalid_model_falls_back_to_chat(self):
        inp = DeepSeekBlock.Input(**_make_input(model="unknown-model"))
        assert inp.model == DeepSeekModel.CHAT

    def test_valid_chat_model_preserved(self):
        inp = DeepSeekBlock.Input(**_make_input(model="deepseek-chat"))
        assert inp.model == DeepSeekModel.CHAT

    def test_valid_reasoner_model_preserved(self):
        inp = DeepSeekBlock.Input(**_make_input(model="deepseek-reasoner"))
        assert inp.model == DeepSeekModel.REASONER

    def test_default_model_when_omitted(self):
        inp = DeepSeekBlock.Input(**_make_input())
        assert inp.model == DeepSeekModel.CHAT

    def test_validate_data_sanitizes_invalid_model(self):
        data = _make_input(model="invalid-model-name")
        error = DeepSeekBlock.Input.validate_data(data)
        assert error is None
        assert data["model"] == DeepSeekModel.CHAT.value

    def test_max_tokens_negative_validation_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeepSeekBlock.Input(**_make_input(max_tokens=-1))

    def test_temperature_out_of_bounds_validation_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeepSeekBlock.Input(**_make_input(temperature=-0.1))

        with pytest.raises(ValidationError):
            DeepSeekBlock.Input(**_make_input(temperature=2.1))


class TestDeepSeekBlockExecution:
    """Tests for DeepSeek block async execution and API mocking."""

    @pytest.mark.asyncio
    async def test_deepseek_chat_execution(self):
        block = DeepSeekBlock()
        mock_credentials = APIKeyCredentials(
            id="test-id",
            provider="deepseek",
            api_key=SecretStr("sk-test-key"),
            title="Test Key",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "This is a response from DeepSeek-V3."
        mock_choice.message.reasoning_content = None

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            result = await block.call_deepseek(
                credentials=mock_credentials,
                model=DeepSeekModel.CHAT,
                prompt="Tell me a joke",
                system_prompt="Be funny",
                temperature=0.7,
                max_tokens=100,
            )

            assert result["response"] == "This is a response from DeepSeek-V3."
            assert result["reasoning_content"] == ""
            assert block.execution_stats.input_token_count == 15
            assert block.execution_stats.output_token_count == 30

            mock_client.chat.completions.create.assert_called_once_with(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Be funny"},
                    {"role": "user", "content": "Tell me a joke"},
                ],
                max_tokens=100,
                stream=False,
                temperature=0.7,
            )

    @pytest.mark.asyncio
    async def test_deepseek_reasoner_execution(self):
        block = DeepSeekBlock()
        mock_credentials = APIKeyCredentials(
            id="test-id",
            provider="deepseek",
            api_key=SecretStr("sk-test-key"),
            title="Test Key",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "Final Answer: 42"
        mock_choice.message.reasoning_content = (
            "Thinking Step 1: Analyze problem. Step 2: Compute."
        )

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 25
        mock_usage.completion_tokens = 80

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            result = await block.call_deepseek(
                credentials=mock_credentials,
                model=DeepSeekModel.REASONER,
                prompt="Solve the riddle",
                system_prompt="",
                temperature=0.9,  # Should be ignored for reasoner
            )

            assert result["response"] == "Final Answer: 42"
            assert (
                result["reasoning_content"]
                == "Thinking Step 1: Analyze problem. Step 2: Compute."
            )
            assert block.execution_stats.input_token_count == 25
            assert block.execution_stats.output_token_count == 80

            # Verify temperature was not passed for reasoner model
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_deepseek_json_mode_with_json_in_prompt(self):
        block = DeepSeekBlock()
        mock_credentials = APIKeyCredentials(
            id="test-id",
            provider="deepseek",
            api_key=SecretStr("sk-test-key"),
            title="Test Key",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = '{"status": "ok"}'
        mock_choice.message.reasoning_content = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            result = await block.call_deepseek(
                credentials=mock_credentials,
                model=DeepSeekModel.CHAT,
                prompt="Output JSON object",
                json_mode=True,
            )

            assert result["response"] == '{"status": "ok"}'
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}
            assert call_kwargs["messages"] == [
                {"role": "user", "content": "Output JSON object"}
            ]

    @pytest.mark.asyncio
    async def test_deepseek_json_mode_without_json_in_prompt_appends_instruction(self):
        block = DeepSeekBlock()
        mock_credentials = APIKeyCredentials(
            id="test-id",
            provider="deepseek",
            api_key=SecretStr("sk-test-key"),
            title="Test Key",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = '{"count": 5}'
        mock_choice.message.reasoning_content = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            result = await block.call_deepseek(
                credentials=mock_credentials,
                model=DeepSeekModel.CHAT,
                prompt="List top 5 fruits",
                system_prompt="You are a helpful assistant.",
                json_mode=True,
            )

            assert result["response"] == '{"count": 5}'
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}
            assert call_kwargs["messages"] == [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.\nRespond with a valid JSON object.",
                },
                {"role": "user", "content": "List top 5 fruits"},
            ]

    @pytest.mark.asyncio
    async def test_deepseek_streaming_execution(self):
        block = DeepSeekBlock()
        mock_credentials = APIKeyCredentials(
            id="test-id",
            provider="deepseek",
            api_key=SecretStr("sk-test-key"),
            title="Test Key",
        )

        # Mock streaming chunks
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [
            MagicMock(delta=MagicMock(content="Hello", reasoning_content="Thinking..."))
        ]

        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.choices = [
            MagicMock(delta=MagicMock(content=" world!", reasoning_content=" Done."))
        ]

        chunk3 = MagicMock()
        chunk3.choices = []
        chunk3.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        async def async_chunk_generator():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=async_chunk_generator()
            )
            mock_openai_cls.return_value = mock_client

            result = await block.call_deepseek(
                credentials=mock_credentials,
                model=DeepSeekModel.CHAT,
                prompt="Stream me a greeting",
                stream=True,
            )

            assert result["response"] == "Hello world!"
            assert result["reasoning_content"] == "Thinking... Done."
            assert block.execution_stats.input_token_count == 10
            assert block.execution_stats.output_token_count == 20

    @pytest.mark.asyncio
    async def test_deepseek_run_method_and_error_handling(self):
        block = DeepSeekBlock()
        input_data = DeepSeekBlock.Input(
            **_make_input(prompt="Test prompt", model="deepseek-chat")
        )

        # Success case
        with patch.object(
            block,
            "call_deepseek",
            AsyncMock(
                return_value={
                    "response": "Success result",
                    "reasoning_content": "Reasoning steps",
                }
            ),
        ):
            outputs = [
                out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)
            ]
            assert outputs == [
                ("response", "Success result"),
                ("reasoning_content", "Reasoning steps"),
            ]

        # Error case
        with patch.object(
            block,
            "call_deepseek",
            AsyncMock(side_effect=RuntimeError("API Connection Timeout")),
        ):
            with pytest.raises(
                BlockExecutionError, match="Error calling DeepSeek: API Connection Timeout"
            ):
                _ = [
                    out
                    async for out in block.run(input_data, credentials=TEST_CREDENTIALS)
                ]


