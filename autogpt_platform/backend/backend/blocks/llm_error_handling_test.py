import json
from typing import cast
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai
import pytest

from backend.blocks import llm
from backend.data.llm_registry.llm_models import LLMModel
from backend.util.llm.prompt_refusal import PROMPT_REFUSED_MESSAGE, PromptRefusedError
from backend.util.llm.provider_billing import PROVIDER_UNAVAILABLE_MESSAGE


@pytest.mark.parametrize(
    "model,status,body,attempts,guided",
    [
        (
            LLMModel.GPT4O_MINI,
            404,
            {"error": {"code": "model_not_found", "message": "Missing resource"}},
            1,
            True,
        ),
        (
            LLMModel.GPT4O_MINI,
            400,
            {"error": {"message": "The model `retired-model` does not exist"}},
            1,
            True,
        ),
        (
            LLMModel.CLAUDE_4_5_HAIKU,
            404,
            {"error": {"type": "not_found_error", "message": "model: old-claude"}},
            1,
            True,
        ),
        (
            LLMModel.CLAUDE_4_5_HAIKU,
            400,
            {"error": {"type": "invalid_request_error", "message": "invalid model ID"}},
            1,
            True,
        ),
        (
            LLMModel.GPT4O_MINI,
            503,
            {"error": {"code": "model_not_found", "message": "Model unavailable"}},
            3,
            False,
        ),
        (
            LLMModel.GPT4O_MINI,
            400,
            {"error": {"message": "Invalid model response format"}},
            3,
            False,
        ),
        (
            LLMModel.GPT4O_MINI,
            404,
            {"error": {"message": "The model-config resource does not exist"}},
            3,
            False,
        ),
        (
            LLMModel.CLAUDE_4_5_HAIKU,
            400,
            {"error": {"type": "invalid_request_error", "message": ["invalid model"]}},
            3,
            False,
        ),
        (
            LLMModel.GPT4O_MINI,
            400,
            {"error": {"message": "Bad input", "metadata": {"raw": "invalid model"}}},
            3,
            False,
        ),
    ],
)
async def test_provider_http_errors_control_retries(
    model: LLMModel, status: int, body: dict, attempts: int, guided: bool
):
    _, _, requests, message = await _run_error(status, body, model)

    assert len(requests) == attempts
    assert ("update the block's model configuration" in message) == guided
    if guided:
        assert model.value in message
        assert model.metadata.provider_name in message


async def test_invalid_model_guidance_does_not_echo_provider_payload():
    _, _, _, message = await _run_error(
        404,
        {
            "error": {
                "code": "model_not_found",
                "message": "Invalid model ID; echoed prompt: private-prompt-canary",
                "debug": "private-debug-canary",
            }
        },
        LLMModel.GPT4O_MINI,
    )

    assert "private-prompt-canary" not in message
    assert "private-debug-canary" not in message
    assert "https://developers.openai.com/api/docs/models" in message


@pytest.mark.parametrize("max_tokens", [1, 2, None])
async def test_context_retry_keeps_outgoing_token_budget_positive(
    max_tokens: int | None,
):
    model = LLMModel.CLAUDE_4_5_HAIKU
    _, data, requests, message = await _run_error(
        400,
        {
            "error": {
                "type": "invalid_request_error",
                "message": "This model's maximum context length was exceeded",
            }
        },
        model,
        max_tokens=max_tokens,
        retry=5,
    )

    expected = [max_tokens or model.max_output_tokens]
    for _ in range(4):
        expected.append(max(1, int(expected[-1] * 0.85)))
    assert [r["max_tokens"] for r in requests] == expected
    assert data.max_tokens == max(1, int(expected[-1] * 0.85))
    assert "update the block's model configuration" not in message


async def test_model_failure_preserves_spend_from_an_earlier_attempt():
    block = llm.AIStructuredResponseGeneratorBlock()
    response = httpx.Response(
        404, request=httpx.Request("POST", "https://api.openai.com")
    )
    error = openai.NotFoundError(
        "Missing model", response=response, body={"code": "model_not_found"}
    )
    call = AsyncMock(
        side_effect=[
            llm.LLMResponse(
                raw_response={},
                prompt=[],
                response="not valid JSON",
                tool_calls=None,
                prompt_tokens=10,
                completion_tokens=2,
                provider_cost=0.01,
            ),
            error,
        ]
    )
    data = block.Input(
        prompt="Test",
        model=LLMModel.GPT4O_MINI,
        expected_format={"answer": "answer"},
        retry=3,
    )

    with patch.object(block, "llm_call", call), pytest.raises(RuntimeError):
        async for _ in block.run(data, credentials=llm.TEST_CREDENTIALS):
            pass

    assert call.await_count == 2
    assert block.execution_stats.provider_cost == pytest.approx(0.01)
    assert block.execution_stats.provider_cost_type == "cost_usd"


async def test_provider_outage_takes_precedence_over_model_message():
    with (
        patch("backend.blocks.llm._is_platform_credential", return_value=True),
        patch("backend.blocks.llm.report_provider_out_of_credits") as report,
    ):
        _, _, requests, message = await _run_error(
            402,
            {"error": {"message": "Invalid model ID", "code": "billing_error"}},
            LLMModel.GPT4O_MINI,
        )

    assert len(requests) == 1
    assert message == PROVIDER_UNAVAILABLE_MESSAGE
    report.assert_called_once()


async def test_prompt_refusal_takes_precedence_over_model_message():
    _, _, requests, message = await _run_error(
        400,
        {"error": {"message": "Invalid model ID", "code": "invalid_prompt"}},
        LLMModel.GPT4O_MINI,
        error_type=PromptRefusedError,
    )

    assert len(requests) == 1
    assert message == PROMPT_REFUSED_MESSAGE


async def _run_error(
    status: int,
    body: dict,
    model: LLMModel,
    *,
    max_tokens: int | None = 100,
    retry: int = 3,
    error_type: type[Exception] = RuntimeError,
):
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(status, json=body)

    credentials = llm.TEST_CREDENTIALS.model_copy(
        update={"provider": model.metadata.provider}
    )
    data = llm.AIStructuredResponseGeneratorBlock.Input(
        prompt="Test",
        model=model,
        credentials=cast(
            llm.AICredentials,
            {"id": credentials.id, "provider": credentials.provider, "type": "api_key"},
        ),
        expected_format={},
        max_tokens=max_tokens,
        retry=retry,
    )
    block = llm.AIStructuredResponseGeneratorBlock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        if model.metadata.provider == "anthropic":
            client = anthropic.AsyncAnthropic(
                api_key="test-only", http_client=http, max_retries=0
            )
            target = "backend.util.llm.providers.anthropic.AsyncAnthropic"
        else:
            client = openai.AsyncOpenAI(
                api_key="test-only", http_client=http, max_retries=0
            )
            target = "backend.util.llm.providers.openai.AsyncOpenAI"
        with patch(target, return_value=client), pytest.raises(error_type) as caught:
            async for _ in block.run(data, credentials=credentials):
                pass
    return block, data, requests, str(caught.value)
