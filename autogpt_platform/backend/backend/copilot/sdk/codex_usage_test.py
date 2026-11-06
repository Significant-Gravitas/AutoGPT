"""Reported Codex usage survives an interrupted response, once per call."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
import httpx
import pytest

from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.testing.codex_lease import codex_route
from backend.copilot.sdk.model_gateway.usage import GatewayUsage
from backend.copilot.sdk.service import _gateway_usage

USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 20,
    "output_tokens_details": {"reasoning_tokens": 5},
    "total_tokens": 120,
}
PAYLOAD = {
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [
        {"name": "lookup", "description": "Look up", "input_schema": {"type": "object"}}
    ],
}


@pytest.mark.parametrize(
    "ending,calls",
    [
        ("failed", 1),
        ("disconnect", 1),
        ("timeout", 1),
        ("cancel", 1),
        ("success", 2),
        ("failed_with_usage", 2),
        ("incomplete", 2),
        ("early_failure", 0),
    ],
)
async def test_reported_usage_survives_the_turns_ending(ending, calls):
    async with usage_gateway(ending) as gateway:
        await run_conversation(gateway, cancel_at_tool=ending == "cancel")
    assert _gateway_usage(gateway) == (expected_usage(calls) if calls else None)


async def run_conversation(gateway: ModelGateway, *, cancel_at_tool: bool = False):
    async with aiohttp.ClientSession() as client:

        async def send(body):
            async with client.post(
                f"{gateway.base_url}/v1/messages",
                json=body,
                headers={"x-api-key": gateway.auth_token},
            ) as response:
                return response.status, await response.json()

        status, first = await send(PAYLOAD)
        if status != 200 or cancel_at_tool:
            return
        call = next(b for b in first["content"] if b["type"] == "tool_use")
        await send(
            {
                **PAYLOAD,
                "messages": [
                    *PAYLOAD["messages"],
                    {"role": "assistant", "content": first["content"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": "found",
                            }
                        ],
                    },
                ],
            }
        )


@asynccontextmanager
async def usage_gateway(ending: str) -> AsyncIterator[ModelGateway]:
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        if ending == "early_failure" or (calls == 2 and ending == "disconnect"):
            raise httpx.ReadError("provider disconnected")
        if calls == 2 and ending == "timeout":
            raise httpx.ReadTimeout("provider timed out")
        tool = calls == 1
        success = ending == "success"
        kind = (
            "response.completed"
            if tool or success
            else (
                "response.incomplete" if ending == "incomplete" else "response.failed"
            )
        )
        usage = (
            USAGE
            if tool or success or ending in ("failed_with_usage", "incomplete")
            else None
        )
        events = [
            {
                "type": "response.created",
                "response": {"id": f"r{calls}", "usage": usage},
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
                        "call_id": "call-1",
                    },
                }
            )
        events.append({"type": kind, "response": {"id": f"r{calls}", "usage": usage}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    route = codex_route("gpt-5.6-sol")
    async with httpx.AsyncClient(
        base_url="https://codex.test", transport=httpx.MockTransport(respond)
    ) as client:
        adapter = CodexResponses(route, client=client)
        async with ModelGateway(route, upstream=adapter) as gateway:
            yield gateway


def expected_usage(calls: int) -> GatewayUsage:
    return GatewayUsage(
        input_tokens=60 * calls,
        cache_read_tokens=40 * calls,
        output_tokens=20 * calls,
        reasoning_tokens=5 * calls,
    )


async def test_cancelling_an_unfinished_response_retains_its_usage_snapshot():
    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self):
            for event in [
                {
                    "type": "response.in_progress",
                    "response": {"id": "partial", "usage": USAGE},
                },
                {"type": "response.output_text.delta", "delta": "partial"},
            ]:
                yield f"data: {json.dumps(event)}\n\n".encode()
            await asyncio.Event().wait()

        async def aclose(self):
            pass

    route = codex_route("gpt-5.6-sol")
    async with httpx.AsyncClient(
        base_url="https://codex.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Interrupted()
            )
        ),
    ) as client:
        adapter = CodexResponses(route, client=client)
        async with ModelGateway(route, upstream=adapter) as gateway:
            async with aiohttp.ClientSession() as incoming:
                async with incoming.post(
                    f"{gateway.base_url}/v1/messages",
                    json={**PAYLOAD, "stream": True},
                    headers={"x-api-key": gateway.auth_token},
                ) as response:
                    await response.content.readuntil(b"\n\n")
        assert _gateway_usage(gateway) == expected_usage(1)
