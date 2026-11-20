"""Exercise the native Responses caller through the real SDK's HTTP transport."""

import json
from copy import deepcopy

import httpx
import openai
import pytest
import pytest_asyncio

from backend.util.llm import providers


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_mode", ["sync", "flex"])
@pytest.mark.parametrize(
    "strict_fields",
    [{"strict": False}, {"strict": True}, {}, {"strict": None}],
    ids=["false", "true", "missing", "null"],
)
async def test_responses_serializes_function_strictness(
    responses_requests, search_function, execution_mode, strict_fields
):
    function = {**search_function, **strict_fields}
    tools = [
        {"type": "function", "function": function},
        {"type": "web_search_preview"},
    ]
    original = deepcopy(tools)

    result = await _call_responses(tools, execution_mode)

    assert len(responses_requests) == 1
    request = responses_requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/responses"
    payload = json.loads(request.content)
    expected_function = {"type": "function", **function}
    if strict_fields.get("strict") is None:
        expected_function.pop("strict", None)
    else:
        assert payload["tools"][0]["strict"] is strict_fields["strict"]
    assert payload["tools"] == [expected_function, {"type": "web_search_preview"}]
    assert tools == original
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["parallel_tool_calls"] is False
    assert payload["max_output_tokens"] == 32
    assert "temperature" not in payload
    if execution_mode == "flex":
        assert payload["service_tier"] == "flex"
    else:
        assert "service_tier" not in payload
    assert result.content == "ok"
    assert (result.prompt_tokens, result.completion_tokens) == (2, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [None, []])
async def test_responses_omits_absent_tools(responses_requests, tools):
    await _call_responses(tools)

    assert len(responses_requests) == 1
    assert "tools" not in json.loads(responses_requests[0].content)


@pytest.mark.asyncio
@pytest.mark.parametrize("function", [{}, {"name": ""}, {"name": None}])
async def test_responses_rejects_missing_function_name_before_request(
    responses_requests, function
):
    tools = [{"type": "function", "function": {"strict": False, **function}}]
    original = deepcopy(tools)

    with pytest.raises(ValueError, match="missing required 'name' field"):
        await _call_responses(tools)

    assert responses_requests == []
    assert tools == original


async def _call_responses(tools, execution_mode="sync"):
    return await providers.call_provider(
        provider="openai",
        model="gpt-4.1",
        api_key="test-not-a-real-key",
        messages=[{"role": "user", "content": "Search"}],
        max_tokens=32,
        tools=tools,
        parallel_tool_calls=False,
        execution_mode=execution_mode,
    )


@pytest.fixture
def search_function():
    return {
        "name": "search",
        "description": "Search with optional filters",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {"strict": {"type": "boolean", "default": False}},
                    "additionalProperties": True,
                },
            },
            "required": ["query"],
            "additionalProperties": True,
        },
    }


@pytest_asyncio.fixture
async def responses_requests(monkeypatch):
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4.1",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        )

    async with openai.AsyncOpenAI(
        api_key="test-not-a-real-key",
        base_url="https://responses.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as client:
        monkeypatch.setattr(providers.openai, "AsyncOpenAI", lambda **kwargs: client)
        yield requests
