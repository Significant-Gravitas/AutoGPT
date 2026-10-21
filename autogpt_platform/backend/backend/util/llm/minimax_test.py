"""MiniMax catalog choices and serialized Chat Completions requests."""

import json
from copy import deepcopy
from unittest.mock import patch

import httpx
import openai
import pytest
from pydantic import TypeAdapter

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.llm_models import LLMModel
from backend.util.llm.providers import call_provider


def test_minimax_picker_preserves_saved_models():
    adapter = TypeAdapter(LLMModel)
    schema = adapter.json_schema()
    choices = schema["llm_model_metadata"]
    assert {name for name in choices if choices[name]["provider"] == "minimax"} == {
        "MiniMax-M3.1-Flash",
        "MiniMax-M3",
    }
    assert [
        model.slug
        for model in get_catalog().models
        if model.provider == "minimax" and model.visibility == "GA"
    ] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert adapter.validate_python("MiniMax-M2.7").value == "MiniMax-M2.7"
    assert "MiniMax-M2.7" in schema["enum"]
    models = {model.slug: model for model in get_catalog().models}
    flash = models["MiniMax-M3.1-Flash"]
    assert flash.context_window == 1_000_000
    assert flash.max_output_tokens == 524_288
    assert flash.supports_tools and flash.supports_reasoning
    assert flash.cost is not None
    assert flash.cost.input_credits_per_1m == 22.5
    assert flash.cost.output_credits_per_1m == 90.0
    assert flash.cost.cache_read_credits_per_1m == 2.25
    assert flash.cost.cache_creation_credits_per_1m is None
    assert models["MiniMax-M2.7"].is_enabled


@pytest.mark.parametrize(
    "provider,model,endpoint",
    [
        ("minimax", model, "https://api.minimax.io/v1/chat/completions")
        for model in [
            "MiniMax-M3.1-Flash",
            "MiniMax-M3",
            "MiniMax-M2.7",
            "custom-model",
        ]
    ]
    + [
        (
            "llama_api",
            "saved-llama-model",
            "https://api.llama.com/compat/v1/chat/completions",
        ),
        (
            "open_router",
            "saved/custom-model",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    ],
)
@pytest.mark.parametrize("execution_mode", ["sync", "flex"])
@pytest.mark.asyncio
async def test_minimax_serialized_request(provider, model, endpoint, execution_mode):
    messages = [
        {"role": "system", "content": "Keep the conversation history."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should inspect the image.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "inspect_image", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "A landscape."},
    ]
    original_messages = deepcopy(messages)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_image",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "A landscape.",
                            (
                                "reasoning_content"
                                if provider == "minimax"
                                else "reasoning"
                            ): "The tool confirms the image.",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client_class = openai.AsyncOpenAI
        with patch.object(
            openai,
            "AsyncOpenAI",
            side_effect=lambda **kwargs: client_class(**kwargs, http_client=http),
        ):
            response = await call_provider(
                provider=provider,
                model=model,
                api_key="offline-test-key",
                messages=messages,
                max_tokens=256,
                tools=tools,
                execution_mode=execution_mode,
            )
    assert len(requests) == 1
    assert str(requests[0].url) == endpoint
    body = json.loads(requests[0].content)
    expected = {
        "model": model,
        "messages": original_messages,
        "max_tokens": 256,
        "tools": tools,
    }
    if provider == "open_router":
        expected["usage"] = {"include": True}
        if execution_mode == "flex":
            expected["service_tier"] = "flex"
    assert body == expected
    assert messages == original_messages
    assert response.content == "A landscape."
    assert response.reasoning == "The tool confirms the image."
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 4
