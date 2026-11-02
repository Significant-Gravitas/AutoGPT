"""FlexOpenAIClient injects service_tier=flex on every underlying call."""

from unittest.mock import AsyncMock

import pytest
from graphiti_core.llm_client import LLMConfig
from pydantic import BaseModel

from backend.copilot.graphiti.flex_client import FlexOpenAIClient


class _DummyResponseModel(BaseModel):
    answer: str


def _make_client_with_stub_openai() -> tuple[FlexOpenAIClient, AsyncMock, AsyncMock]:
    """Build a FlexOpenAIClient with its OpenAI client SDK swapped for mocks."""
    config = LLMConfig(api_key="test-key", model="gpt-4.1-mini")
    client = FlexOpenAIClient(config=config)

    parse_mock = AsyncMock(return_value=object())
    chat_create_mock = AsyncMock(return_value=object())

    client.client.responses.parse = parse_mock  # type: ignore[assignment]
    client.client.chat.completions.create = chat_create_mock  # type: ignore[assignment]

    return client, parse_mock, chat_create_mock


@pytest.mark.asyncio
async def test_structured_completion_sends_service_tier_in_extra_body():
    client, parse_mock, _ = _make_client_with_stub_openai()

    await client._create_structured_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=128,
        response_model=_DummyResponseModel,
    )

    parse_mock.assert_awaited_once()
    kwargs = parse_mock.await_args.kwargs
    assert kwargs["extra_body"] == {"service_tier": "flex"}
    assert kwargs["text_format"] is _DummyResponseModel


@pytest.mark.asyncio
async def test_completion_sends_service_tier_in_extra_body():
    client, _, chat_create_mock = _make_client_with_stub_openai()

    await client._create_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=128,
    )

    chat_create_mock.assert_awaited_once()
    kwargs = chat_create_mock.await_args.kwargs
    assert kwargs["extra_body"] == {"service_tier": "flex"}
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_reasoning_model_skips_temperature_keeps_flex():
    """gpt-5 family doesn't accept temperature, but still gets the flex tag."""
    client, parse_mock, chat_create_mock = _make_client_with_stub_openai()

    await client._create_structured_completion(
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=128,
        response_model=_DummyResponseModel,
        reasoning="medium",
        verbosity="low",
    )

    parse_kwargs = parse_mock.await_args.kwargs
    assert "temperature" not in parse_kwargs
    assert parse_kwargs["reasoning"] == {"effort": "medium"}
    assert parse_kwargs["text"] == {"verbosity": "low"}
    assert parse_kwargs["extra_body"] == {"service_tier": "flex"}

    await client._create_completion(
        model="gpt-5-thinking",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=128,
    )
    chat_kwargs = chat_create_mock.await_args.kwargs
    assert chat_kwargs["temperature"] is None
    assert chat_kwargs["extra_body"] == {"service_tier": "flex"}
