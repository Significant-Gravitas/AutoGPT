import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from backend.copilot.billing_e2e.provider import FakeProvider
from backend.copilot.billing_e2e.wire import TOOL_NAME
from backend.copilot.provider_failure import ProviderFailureKind
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.stream import StreamItem, collect_message
from backend.copilot.sdk.model_gateway.testing import cli_requests
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_backend_client,
)
from backend.copilot.sdk.model_gateway.testing.wire import CODEX_OVERFLOW
from backend.copilot.sdk.model_gateway.usage import GatewayUsage
from backend.data.llm_registry.llm_models import LLMModel

_MODEL = LLMModel.GPT5_6_TERRA.value
_ROUTE = codex_route(_MODEL, context_window=272_000)


async def _drain(items: AsyncIterator[StreamItem]) -> list[StreamItem]:
    return [item async for item in items]


async def _stream_against(reply: CannedReply, body: dict[str, Any] | None = None):
    async with CannedUpstream(reply) as upstream:
        async with fake_backend_client(f"{upstream.url}/codex") as client:
            adapter = CodexResponses(_ROUTE, client=client)
            try:
                return (
                    await _drain(adapter.stream(body or cli_requests.first_request())),
                    adapter,
                    upstream,
                )
            except GatewayError as error:
                return error, adapter, upstream


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _billing_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tool = {
        "name": TOOL_NAME,
        "description": "Look a fact up.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    return {
        "model": _MODEL,
        "system": "You are a test.",
        "messages": messages,
        "tools": [tool],
        "stream": True,
    }


async def test_a_tool_round_trip_against_the_billing_suite_s_codex_backend():
    async with FakeProvider() as provider:
        async with fake_backend_client(f"{provider.url}/codex") as client:
            adapter = CodexResponses(_ROUTE, client=client)
            ask = [_user("Please [use-tool] for me.")]
            first = await collect_message(adapter.stream(_billing_body(ask)))
            (call,) = first["content"]
            assert call["type"] == "tool_use" and call["name"] == TOOL_NAME
            assert call["input"] == {"query": "billing"}
            result = {
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": "no such tool",
                "is_error": True,
            }
            history = [
                *ask,
                {"role": "assistant", "content": [call]},
                {"role": "user", "content": [result]},
            ]
            second = await collect_message(adapter.stream(_billing_body(history)))

    assert first["stop_reason"] == "tool_use"
    assert second["content"] == [
        {"type": "text", "text": "Here is what the lookup found."}
    ]
    assert second["stop_reason"] == "end_turn"
    assert provider.unexpected == []
    billed = [c.usage for c in provider.calls]
    assert len(billed) == 2
    assert second["usage"] == {
        "input_tokens": billed[1].input_tokens,
        "output_tokens": billed[1].output_tokens,
        "cache_read_input_tokens": billed[1].cache_read_tokens,
        "cache_creation_input_tokens": 0,
    }


async def test_a_context_error_before_the_stream_is_the_cli_s_overflow_wording():
    reply = CannedReply.of_json(
        {"error": {**CODEX_OVERFLOW, "type": "invalid_request_error"}}, status=400
    )
    error, _, _ = await _stream_against(reply)
    assert isinstance(error, GatewayError)
    assert error.status == 400
    assert error.message.startswith("prompt is too long: ")
    assert error.message.endswith(" tokens > 272000 maximum")
    assert error.failure is None


async def test_a_context_failure_inside_the_stream_is_still_an_http_error():
    response = {"id": "r", "status": "failed", "error": CODEX_OVERFLOW}
    reply = CannedReply.of_sse(
        [
            {"type": "response.created", "response": {"id": "r"}},
            {"type": "response.failed", "response": response},
        ]
    )
    error, _, _ = await _stream_against(reply)
    assert isinstance(error, GatewayError)
    assert error.status == 400
    assert error.message.endswith(" tokens > 272000 maximum")


async def test_a_spent_quota_is_a_429_with_its_reset_and_no_retries():
    body = {
        "error": {
            "type": "usage_limit_reached",
            "message": "You've hit your usage limit.",
            "resets_in_seconds": 1800,
        }
    }
    error, _, _ = await _stream_against(CannedReply.of_json(body, status=429))
    assert isinstance(error, GatewayError)
    assert (error.status, error.error_type) == (429, "rate_limit_error")
    assert error.headers == {"retry-after": "1800", "x-should-retry": "false"}
    assert error.failure is not None
    assert error.failure.kind == ProviderFailureKind.USAGE_LIMIT
    assert error.failure.auth_provider == "codex"


async def test_a_reset_time_becomes_retry_after_seconds():
    resets_at = int(time.time()) + 600
    body = {
        "error": {
            "type": "usage_limit_reached",
            "message": "limit",
            "resets_at": resets_at,
        }
    }
    error, _, _ = await _stream_against(CannedReply.of_json(body, status=429))
    assert isinstance(error, GatewayError)
    assert 590 <= int(error.headers["retry-after"]) <= 600


async def test_a_transient_rate_limit_passes_on_the_backend_s_retry_after():
    body = {"error": {"type": "rate_limit_exceeded", "message": "Rate limit reached."}}
    error, _, _ = await _stream_against(
        CannedReply.of_json(body, status=429, **{"retry-after": "4"})
    )
    assert isinstance(error, GatewayError)
    assert error.headers == {"retry-after": "4"}


async def test_an_in_stream_rate_limit_waits_as_long_as_its_message_says():
    error_body = {
        "code": "rate_limit_exceeded",
        "message": "Rate limit reached for gpt-5 on tokens per min. Please try"
        " again in 1.898s.",
    }
    response = {"id": "r", "status": "failed", "error": error_body}
    error, _, _ = await _stream_against(
        CannedReply.of_sse([{"type": "response.failed", "response": response}])
    )
    assert isinstance(error, GatewayError)
    assert (error.status, error.error_type) == (429, "rate_limit_error")
    assert error.headers == {"retry-after": "2"}


async def test_a_refusal_still_records_the_quota_headers():
    body = {"error": {"type": "usage_limit_reached", "message": "limit"}}
    reply = CannedReply.of_json(
        body,
        status=429,
        **{
            "x-codex-primary-used-percent": "100",
            "x-codex-primary-window-minutes": "300",
            "x-codex-primary-reset-at": "1999999999",
        },
    )
    error, adapter, _ = await _stream_against(reply)
    assert isinstance(error, GatewayError)
    assert adapter.rate_limits is not None
    assert adapter.rate_limits.primary is not None
    assert adapter.rate_limits.primary.used_percent == 100
    assert adapter.rate_limits.primary.resets_at == 1999999999


async def test_an_expired_login_says_so_and_names_the_failure():
    body = {"detail": "Could not validate your credentials"}
    error, _, _ = await _stream_against(CannedReply.of_json(body, status=401))
    assert isinstance(error, GatewayError)
    assert (error.status, error.error_type) == (401, "authentication_error")
    assert error.failure is not None
    assert error.failure.kind == ProviderFailureKind.AUTH_EXPIRED


async def test_a_server_failure_inside_the_stream_is_retryable():
    response = {
        "id": "r",
        "status": "failed",
        "error": {"code": "server_error", "message": "oops"},
    }
    reply = CannedReply.of_sse([{"type": "response.failed", "response": response}])
    error, _, _ = await _stream_against(reply)
    assert isinstance(error, GatewayError)
    assert (error.status, error.error_type) == (500, "api_error")


async def test_quota_headers_are_kept_for_the_platform_to_report():
    reply = CannedReply.of_sse(
        [
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "m"},
            },
            {"type": "response.output_text.delta", "item_id": "m", "delta": "ok"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 1}},
            },
        ]
    )
    reply.headers.update(
        {
            "x-codex-primary-used-percent": "42",
            "x-codex-primary-window-minutes": "300",
            "x-codex-plan-type": "plus",
        }
    )
    items, adapter, _ = await _stream_against(reply)
    assert isinstance(items, list)
    assert adapter.rate_limits is not None
    assert adapter.rate_limits.plan_type == "plus"
    assert adapter.rate_limits.primary is not None
    assert adapter.rate_limits.primary.used_percent == 42
    assert items[-2].usage == GatewayUsage(input_tokens=5, output_tokens=1)


async def test_a_stream_that_never_completes_is_an_error():
    reply = CannedReply.of_sse(
        [
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "m"},
            },
            {"type": "response.output_text.delta", "item_id": "m", "delta": "partial"},
        ]
    )
    error, _, _ = await _stream_against(reply)
    assert isinstance(error, GatewayError)
    assert (error.status, error.message) == (502, "ChatGPT ended the response early")


async def test_each_call_sends_the_full_history_with_nothing_stored():
    reply = CannedReply.of_sse(
        [
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
            }
        ]
    )
    _, _, upstream = await _stream_against(reply, cli_requests.continuation_request())
    (call,) = upstream.calls
    assert call.path == "/codex/responses"
    assert call.body["store"] is False
    assert [item["type"] for item in call.body["input"]][-3:] == [
        "function_call_output",
        "function_call_output",
        "message",
    ]


async def test_the_client_is_hardened_and_allows_operator_egress_proxies():
    adapter = CodexResponses(codex_route(_MODEL))
    try:
        assert not adapter._client.follow_redirects
        assert adapter._client.trust_env
    finally:
        await adapter.aclose()


def test_the_adapter_refuses_a_route_without_a_lease():
    route = GatewayRoute(
        adapter="openai_chat", model="m", base_url="http://127.0.0.1:1/v1"
    )
    with pytest.raises(ValueError, match="credential lease"):
        CodexResponses(route)
