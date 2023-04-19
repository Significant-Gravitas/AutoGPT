from unittest.mock import MagicMock, patch

import pytest
from openai.error import APIError, RateLimitError

from autogpt.agent.LLM.openai_provider import (
    ChatCompletionModels,
    EmbeddingModels,
    OpenAIProvider,
)


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="fake_api_key")


@pytest.mark.asyncio
async def test_attempt_chat_completion_success(openai_provider):
    with patch("openai.ChatCompletion.acreate") as acreate_mock:
        acreate_mock.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }

        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = await openai_provider.attempt_chat_completion(messages)

        acreate_mock.assert_called_once_with(
            model=openai_provider.chat_completion_model,
            messages=messages,
            temperature=openai_provider.temperature,
            max_tokens=openai_provider.max_tokens,
        )
        assert response == {"choices": [{"message": {"content": "Test response"}}]}


@pytest.mark.skip
@pytest.mark.asyncio
async def test_attempt_chat_completion_rate_limit_error(openai_provider):
    with patch("openai.ChatCompletion.acreate") as acreate_mock:
        acreate_mock.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        acreate_mock.side_effect = RateLimitError("Rate limit exceeded", {}, 429)

        messages = [{"role": "user", "content": "Hello, how are you?"}]
        with pytest.raises(RateLimitError):
            await openai_provider.attempt_chat_completion(messages)

        acreate_mock.assert_called_once_with(
            model=openai_provider.chat_completion_model,
            messages=messages,
            temperature=openai_provider.temperature,
            max_tokens=openai_provider.max_tokens,
        )


@pytest.mark.asyncio
async def test_attempt_chat_completion_api_error(openai_provider):
    with patch("openai.ChatCompletion.acreate") as acreate_mock:
        acreate_mock.side_effect = APIError("API error", {}, 500)

        messages = [{"role": "user", "content": "Hello, how are you?"}]
        with pytest.raises(APIError):
            await openai_provider.attempt_chat_completion(messages)

        acreate_mock.assert_called_once_with(
            model=openai_provider.chat_completion_model,
            messages=messages,
            temperature=openai_provider.temperature,
            max_tokens=openai_provider.max_tokens,
        )


@pytest.mark.asyncio
async def test_create_chat_completion_success(openai_provider: OpenAIProvider) -> None:
    """Test create_chat_completion()"""
    with patch.object(openai_provider, "attempt_chat_completion") as attempt_mock:

        class Message:
            """Mock message class"""

            def __init__(self):
                self.message = {"content": "Test response"}

        class Response:
            """Mock response class"""

            def __init__(self):
                self.choices = [Message()]

        attempt_mock.return_value = Response()

        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = await openai_provider.create_chat_completion(messages)
        print(response)
        assert response == "Test response"
        attempt_mock.assert_called()


@pytest.mark.asyncio
async def test_create_embedding_with_ada_success(openai_provider):
    with patch("openai.Embedding.acreate") as acreate_mock:
        acreate_mock.return_value = {"data": [{"embedding": [0.5, 0.6, 0.7]}]}

        text = "Hello, how are you?"
        response = await openai_provider.create_embedding_with_ada(text)

        acreate_mock.assert_called_once_with(
            input=[text], model=openai_provider.embedding_model.value
        )
        assert response == [0.5, 0.6, 0.7]
        acreate_mock.assert_called()
