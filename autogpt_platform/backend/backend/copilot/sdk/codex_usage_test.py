"""Reported Codex usage survives an interrupted tool loop, once per call."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import SecretStr

from backend.copilot.sdk.codex_compat_gateway import CodexAnthropicGateway
from backend.copilot.sdk.service import _codex_gateway_usage
from backend.data.model import OAuth2Credentials
from backend.integrations.codex.http_session import CodexHttpSession
from backend.integrations.codex.models import CodexDynamicToolResult, CodexTokenUsage

USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 20,
    "output_tokens_details": {"reasoning_tokens": 5},
    "total_tokens": 120,
}
PAYLOAD: dict[str, object] = {
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [
        {"name": "lookup", "description": "Look up", "input_schema": {"type": "object"}}
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["failed", "disconnect", "timeout", "cancel"])
async def test_interrupted_tool_loop_retains_completed_call_usage(ending):
    async with usage_gateway(ending) as gateway:
        await run_conversation(gateway, cancel_at_tool=ending == "cancel")

    assert gateway.result is None
    assert _codex_gateway_usage(gateway) == expected_usage(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["success", "failed_with_usage", "incomplete"])
async def test_response_snapshots_and_final_result_are_counted_once(ending):
    async with usage_gateway(ending) as gateway:
        await run_conversation(gateway)

    assert _codex_gateway_usage(gateway) == expected_usage(2)


@pytest.mark.asyncio
async def test_an_early_failure_has_no_usage():
    async with usage_gateway("early_failure") as gateway:
        await run_conversation(gateway)

    assert _codex_gateway_usage(gateway) is None


@pytest.mark.asyncio
async def test_a_retry_keeps_failed_usage_without_recounting_success():
    async with usage_gateway("failed_then_success") as gateway:
        await run_conversation(gateway)
        await run_conversation(gateway)

    assert _codex_gateway_usage(gateway) == expected_usage(3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending", ["stream_disconnect", "stream_timeout", "stream_cancel", "stream_failed"]
)
async def test_interrupted_stream_retains_usage_and_closes_response(ending):
    stream = _InterruptedStream(ending)
    async with usage_gateway(ending, response_stream=stream) as gateway:
        await run_conversation(
            gateway, cancel_during_response=ending == "stream_cancel"
        )
        assert stream.closed

    assert gateway.result is None
    assert _codex_gateway_usage(gateway) == expected_usage(2)


async def run_conversation(
    gateway: CodexAnthropicGateway,
    *,
    cancel_at_tool: bool = False,
    cancel_during_response: bool = False,
) -> None:
    conversation = gateway._start_conversation(PAYLOAD)
    assert conversation.task is not None
    await asyncio.wait_for(conversation.queue.get(), timeout=2)
    if cancel_at_tool:
        await gateway.close()
        assert conversation.task.cancelled()
        return
    for record in conversation.pending.values():
        record.future.set_result(CodexDynamicToolResult(content="found"))
    if cancel_during_response:
        await asyncio.wait_for(conversation.queue.get(), timeout=2)
        await gateway.close()
        assert conversation.task.cancelled()
        return
    await asyncio.wait_for(conversation.task, timeout=2)


@asynccontextmanager
async def usage_gateway(
    ending: str, *, response_stream: httpx.AsyncByteStream | None = None
) -> AsyncIterator[CodexAnthropicGateway]:
    calls = 0

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return await _provider_response(ending, calls, response_stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        session = CodexHttpSession(
            OAuth2Credentials(
                id="codex-usage-test",
                provider="codex",
                title="ChatGPT",
                access_token=SecretStr("test"),
                scopes=[],
            ),
            turn_timeout_seconds=1 if ending in ("timeout", "stream_timeout") else 5,
            tool_timeout_seconds=5,
            client=AsyncOpenAI(
                api_key="test",
                base_url="https://codex.test",
                http_client=http,
                max_retries=0,
            ),
        )
        async with CodexAnthropicGateway(
            agent_session=session, model="gpt-5.6-sol"
        ) as gateway:
            yield gateway


async def _provider_response(
    ending: str, calls: int, response_stream: httpx.AsyncByteStream | None
) -> httpx.Response:
    if ending == "early_failure" or (calls % 2 == 0 and ending == "disconnect"):
        raise httpx.ReadError("provider disconnected")
    if calls % 2 == 0 and ending == "timeout":
        await asyncio.Event().wait()
    if calls % 2 == 0 and ending.startswith("stream_"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=response_stream or _InterruptedStream(ending),
        )
    events = _response_events(ending, calls)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


def _response_events(ending: str, calls: int) -> list[dict[str, object]]:
    tool = calls % 2 == 1
    success = ending == "success" or (ending == "failed_then_success" and calls > 2)
    kind = "response.completed" if tool or success else "response.failed"
    if not tool and ending == "incomplete":
        kind = "response.incomplete"
    usage = (
        USAGE
        if tool or success or ending in ("failed_with_usage", "incomplete")
        else None
    )
    response = {"id": f"resp-{calls}", "model": "gpt-5.6-sol", "usage": usage}
    events: list[dict[str, object]] = [
        {
            "type": "response.created",
            "response": {**response, "usage": usage if tool or success else None},
        }
    ]
    if tool:
        events.append(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                    "call_id": f"call-{calls}",
                },
            }
        )
    events.append({"type": kind, "response": response})
    return events


class _InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, ending: str):
        self.ending = ending
        self.closed = False

    async def __aiter__(self):
        for event in (
            {
                "type": "response.in_progress",
                "response": {"id": "partial", "usage": USAGE},
            },
            {"type": "response.output_text.delta", "delta": "partial"},
        ):
            yield f"data: {json.dumps(event)}\n\n".encode()
        if self.ending == "stream_disconnect":
            raise httpx.ReadError("stream disconnected")
        if self.ending == "stream_failed":
            failure = {"type": "response.failed", "response": {"usage": USAGE}}
            yield f"data: {json.dumps(failure)}\n\n".encode()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def expected_usage(calls: int) -> CodexTokenUsage:
    return CodexTokenUsage(
        input_tokens=100 * calls,
        cached_input_tokens=40 * calls,
        output_tokens=20 * calls,
        reasoning_output_tokens=5 * calls,
        total_tokens=120 * calls,
    )
