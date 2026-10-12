import json
from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr

from backend.copilot.sdk.model_gateway.adapters.anthropic_passthrough import (
    AnthropicPassthrough,
)
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.route import (
    ApiKeyCredential,
    GatewayRoute,
    NoCredential,
)
from backend.copilot.sdk.model_gateway.stream import StreamItem
from backend.copilot.sdk.model_gateway.testing import cli_requests
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.wire import LLAMA_CPP_OVERFLOW

_KEY = "sk-llama-upstream-0001"
_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "ornith",
            "content": [],
            "usage": {"input_tokens": 15855, "output_tokens": 0},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {"type": "ping"},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hi"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 3},
    },
    {"type": "message_stop"},
]


def _route(
    url: str, *, key: str | None = _KEY, window: int | None = 32_768
) -> GatewayRoute:
    return GatewayRoute(
        adapter="anthropic_passthrough",
        model="ornith-1.5-9b",
        base_url=f"{url}/",
        credential=ApiKeyCredential(api_key=SecretStr(key)) if key else NoCredential(),
        context_window=window,
    )


async def _drain(items: AsyncIterator[StreamItem]) -> list[StreamItem]:
    return [item async for item in items]


async def test_the_request_goes_through_unchanged_but_for_model_billing_line_and_key():
    async with CannedUpstream(CannedReply.of_sse(_EVENTS)) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        body = cli_requests.continuation_request()
        await _drain(adapter.stream(body))
        await adapter.aclose()

    (call,) = upstream.calls
    assert call.path == "/v1/messages"
    expected = {**body, "model": "ornith-1.5-9b", "system": body["system"][1:]}
    assert call.body == expected
    assert call.headers["x-api-key"] == _KEY
    assert call.headers["authorization"] == f"Bearer {_KEY}"
    assert call.headers["anthropic-version"] == "2023-06-01"


async def test_the_stream_is_relayed_frame_for_frame():
    async with CannedUpstream(CannedReply.of_sse(_EVENTS)) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        items = await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()

    assert [item.event for item in items] == _EVENTS
    assert b"".join(item.raw or b"" for item in items) == b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in _EVENTS
    )


async def test_a_server_without_a_key_gets_no_auth_header():
    async with CannedUpstream(CannedReply.of_sse(_EVENTS)) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url, key=None))
        await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert "authorization" not in upstream.calls[0].headers
    assert "x-api-key" not in upstream.calls[0].headers


async def test_llama_cpp_s_overflow_becomes_the_wording_the_cli_compacts_on():
    async with CannedUpstream(
        CannedReply.of_json(LLAMA_CPP_OVERFLOW, status=400)
    ) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert caught.value.status == 400
    assert caught.value.message == "prompt is too long: 40000 tokens > 32768 maximum"


@pytest.mark.parametrize(
    "body",
    [
        {"error": "prompt too long; exceeded max context length by 900 tokens"},
        {
            "error": "Trying to keep the first 33000 tokens when context the overflows."
            " However, the model is loaded with context length of only 32768 tokens"
        },
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "This model's maximum context length is 32768 tokens."
                " However, your request has 33668 input tokens.",
            },
        },
    ],
    ids=["ollama", "lm-studio", "vllm"],
)
async def test_other_servers_overflows_are_rewritten_too(body):
    async with CannedUpstream(CannedReply.of_json(body, status=400)) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert caught.value.message.startswith("prompt is too long: ")
    assert caught.value.message.endswith(" tokens > 32768 maximum")


async def test_an_overflow_sent_as_the_first_stream_event_is_still_an_http_error():
    error = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": LLAMA_CPP_OVERFLOW["error"]["message"],
        },
    }
    async with CannedUpstream(CannedReply.of_sse([error])) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert caught.value.status == 400
    assert caught.value.message.startswith("prompt is too long: 40000 tokens")


async def test_an_error_mid_stream_raises_after_what_was_relayed():
    events = [
        *_EVENTS[:2],
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ]
    async with CannedUpstream(CannedReply.of_sse(events)) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        relayed: list[StreamItem] = []
        with pytest.raises(GatewayError) as caught:
            async for item in adapter.stream(cli_requests.first_request()):
                relayed.append(item)
        await adapter.aclose()
    assert len(relayed) == 2
    assert (caught.value.status, caught.value.error_type) == (529, "overloaded_error")


async def test_errors_keep_status_type_and_retry_hint_and_lose_the_key():
    reply = CannedReply.of_json(
        {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": f"key {_KEY} is throttled",
            },
        },
        status=429,
        **{"retry-after": "3"},
    )
    async with CannedUpstream(reply) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    error = caught.value
    assert (error.status, error.error_type) == (429, "rate_limit_error")
    assert error.headers == {"retry-after": "3"}
    assert _KEY not in error.message


async def test_an_unreachable_server_is_a_502():
    adapter = AnthropicPassthrough(_route("http://127.0.0.1:9"))
    with pytest.raises(GatewayError) as caught:
        await _drain(adapter.stream(cli_requests.first_request()))
    await adapter.aclose()
    assert (caught.value.status, caught.value.message) == (
        502,
        "The model server could not be reached",
    )


async def test_a_non_streaming_request_returns_the_upstream_message():
    message = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    async with CannedUpstream(CannedReply.of_json(message)) as upstream:
        adapter = AnthropicPassthrough(_route(upstream.url))
        body = {**cli_requests.first_request(), "stream": False}
        assert await adapter.complete(body) == message
        await adapter.aclose()
    assert upstream.calls[0].body["stream"] is False
    assert upstream.calls[0].headers["accept"] == "application/json"
