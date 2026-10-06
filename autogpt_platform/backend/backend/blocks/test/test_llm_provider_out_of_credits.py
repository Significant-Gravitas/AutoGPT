"""A platform-owned provider account running dry must not reach the user raw.

OpenRouter answers an empty account with a 402 telling the reader to buy
credits on openrouter.ai. On a platform key that bill is ours, so the block
run shows one plain sentence and we get paged; on the user's own key the
provider's wording is exactly what they need, so it is left alone.
"""

import logging
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

import backend.blocks.llm as llm
from backend.integrations.credentials_store import open_router_credentials

_OPENROUTER_402_MESSAGE = (
    "Insufficient credits. This account never purchased credits. Make sure "
    "your key is on the correct account or org, and if so, purchase more at "
    "https://openrouter.ai/settings/credits"
)


def _openrouter_402() -> openai.APIStatusError:
    body = {"error": {"message": _OPENROUTER_402_MESSAGE, "code": 402}}
    response = httpx.Response(
        402,
        json=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat"),
    )
    return openai.APIStatusError(
        f"Error code: 402 - {body}", response=response, body=body["error"]
    )


def _user_key() -> llm.APIKeyCredentials:
    return open_router_credentials.model_copy(update={"id": "user-owned-key"})


def _input() -> llm.AIStructuredResponseGeneratorBlock.Input:
    return llm.AIStructuredResponseGeneratorBlock.Input(
        prompt="Test",
        expected_format={"key": "desc"},
        model=llm.LLMModel("openai/gpt-oss-120b"),
        credentials=cast(llm.AICredentials, llm.TEST_CREDENTIALS_INPUT),
        retry=3,
    )


@pytest.mark.asyncio
async def test_platform_key_402_shows_platform_message_and_alerts(caplog):
    inner = AsyncMock(side_effect=_openrouter_402())
    with (
        patch.object(llm, "_llm_call", new=inner),
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError) as exc_info,
    ):
        async for _ in llm.AIStructuredResponseGeneratorBlock().run(
            _input(), credentials=open_router_credentials
        ):
            pass

    message = str(exc_info.value)
    assert "temporarily unavailable" in message
    assert "openrouter" not in message.lower()
    assert "credits" not in message.lower()
    # An empty account stays empty: one call, not the whole retry budget.
    assert inner.await_count == 1
    alerts = [
        r
        for r in caplog.records
        if r.getMessage() == "LLM provider account is out of credits"
    ]
    assert len(alerts) == 1
    assert alerts[0].levelno == logging.ERROR
    fields = alerts[0].__dict__["json_fields"]
    assert fields["provider"] == "open_router"
    assert fields["model"] == "openai/gpt-oss-120b"
    assert fields["surface"] == "block"


@pytest.mark.asyncio
async def test_user_key_402_keeps_the_providers_wording(caplog):
    inner = AsyncMock(side_effect=_openrouter_402())
    with (
        patch.object(llm, "_llm_call", new=inner),
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError) as exc_info,
    ):
        async for _ in llm.AIStructuredResponseGeneratorBlock().run(
            _input(), credentials=_user_key()
        ):
            pass

    assert "openrouter.ai/settings/credits" in str(exc_info.value)
    assert not [
        r
        for r in caplog.records
        if r.getMessage() == "LLM provider account is out of credits"
    ]


@pytest.mark.asyncio
async def test_llm_call_raises_platform_error_for_orchestrator_callers():
    """Callers that use ``llm_call`` directly (orchestrator, AI condition)
    surface ``str(exc)`` as the block error, so the seam itself must clean it.
    """
    with (
        patch.object(llm, "_llm_call", new=AsyncMock(side_effect=_openrouter_402())),
        pytest.raises(RuntimeError) as exc_info,
    ):
        await llm.llm_call(
            credentials=open_router_credentials,
            llm_model=llm.LLMModel("openai/gpt-oss-120b"),
            prompt=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )

    assert "temporarily unavailable" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, openai.APIStatusError)


@pytest.mark.asyncio
async def test_non_billing_errors_pass_through_untouched():
    boom = openai.APIStatusError(
        "Error code: 500 - upstream",
        response=httpx.Response(
            500, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat")
        ),
        body=None,
    )
    with (
        patch.object(llm, "_llm_call", new=AsyncMock(side_effect=boom)),
        pytest.raises(openai.APIStatusError),
    ):
        await llm.llm_call(
            credentials=open_router_credentials,
            llm_model=llm.LLMModel("openai/gpt-oss-120b"),
            prompt=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
