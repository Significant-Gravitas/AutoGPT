import json

import httpx
import pytest
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts import Message
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.copilot.graphiti.flex_client import FlexOpenAIClient
from backend.copilot.graphiti.llm_client import GraphitiOpenAIClient


class ExtractedFact(BaseModel):
    fact: str


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [GraphitiOpenAIClient, FlexOpenAIClient])
@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize("model_size", [ModelSize.small, ModelSize.medium])
@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-luna",
        "openai/gpt-6-luna",
        "gpt-4.1-mini",
        "qwen3:8b",
    ],
)
async def test_extraction_request_contract(
    client_type, structured, model_size, model, request_boundary
):
    captured, transport = request_boundary

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    ) as sdk:
        client = client_type(
            config=LLMConfig(model=model, small_model=model), client=sdk
        )
        result = await client.generate_response(
            [Message(role="user", content="Extract JSON facts: Mira lives in Oslo.")],
            response_model=ExtractedFact if structured else None,
            max_tokens=256,
            model_size=model_size,
        )

    assert result == {"fact": "Mira lives in Oslo"}
    assert len(captured) == 1
    _assert_request(captured[0], model, structured, client_type is FlexOpenAIClient)


def _assert_request(body, model, structured, flex):
    assert body["model"] == model
    assert body["temperature"] == 1
    if "gpt-6-luna" in model:
        if structured:
            assert body["reasoning"] == {"effort": "none"}
        else:
            assert body["reasoning_effort"] == "none"
            assert body["max_completion_tokens"] == 256
            assert "max_tokens" not in body
    else:
        assert "reasoning" not in body
        assert "reasoning_effort" not in body
        if not structured:
            assert body["max_tokens"] == 256
    if structured:
        assert body["max_output_tokens"] == 256
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
    else:
        assert body["response_format"] == {"type": "json_object"}
    if flex:
        assert body["service_tier"] == "flex"
    else:
        assert "service_tier" not in body


@pytest.fixture
def request_boundary():
    captured = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        content = '{"fact":"Mira lives in Oslo"}'
        if request.url.path.endswith("/responses"):
            payload = {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": body["model"],
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": content, "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            }
        else:
            payload = {
                "id": "chat_test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            }
        return httpx.Response(200, json=payload)

    return captured, httpx.MockTransport(respond)
