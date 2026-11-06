from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import aiohttp

from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.route import AdapterKind, GatewayRoute
from backend.copilot.sdk.model_gateway.stream import Event
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_backend_client,
)

BODY: Event = {"messages": [{"role": "user", "content": "x" * 400}]}
USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 50,
    "output_tokens_details": {"reasoning_tokens": 30},
}
REFUSAL = "I cannot help with that."


def chunk(delta: Event, finish: str | None = None) -> Event:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def chat_reply(events: list[Event]) -> CannedReply:
    return CannedReply.of_sse(events, named=False, done=True)


def terminal(kind: str = "response.completed", **fields: object) -> Event:
    return {"type": kind, "response": {"usage": USAGE, **fields}}


def text_events() -> list[Event]:
    return [
        {"type": "response.output_text.delta", "item_id": "m", "delta": "answer"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "m",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        },
    ]


def refusal_events() -> list[Event]:
    return [
        {
            "type": "response.refusal.delta",
            "item_id": "m",
            "content_index": 0,
            "delta": REFUSAL,
        },
        {
            "type": "response.refusal.done",
            "item_id": "m",
            "content_index": 0,
            "refusal": REFUSAL,
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "m",
                "content": [{"type": "refusal", "refusal": REFUSAL}],
            },
        },
        terminal(),
    ]


def fragments() -> list[Event]:
    return [
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_",
                        "function": {
                            "name": "mcp__handbook__read_",
                            "arguments": '{"page":',
                        },
                    }
                ]
            }
        ),
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "review",
                        "function": {"name": "doc", "arguments": "1}"},
                    }
                ]
            }
        ),
        chunk({}, "tool_calls"),
    ]


@asynccontextmanager
async def gateway_for(
    adapter: AdapterKind, upstream: CannedUpstream
) -> AsyncIterator[ModelGateway]:
    async with AsyncExitStack() as stack:
        if adapter == "codex_responses":
            route = codex_route("gateway-test-model")
            client = await stack.enter_async_context(
                fake_backend_client(upstream.url + "/codex")
            )
            implementation = CodexResponses(route, client=client)
        else:
            route = GatewayRoute(
                adapter=adapter,
                model="gateway-test-model",
                base_url=upstream.url + "/v1",
            )
            implementation = None
        yield await stack.enter_async_context(
            ModelGateway(route, upstream=implementation)
        )


async def post(gateway: ModelGateway, *, streaming: bool) -> tuple[int, str]:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15)
    ) as session:
        async with session.post(
            gateway.base_url + "/v1/messages",
            headers={"x-api-key": gateway.auth_token},
            json={**BODY, "stream": streaming},
        ) as response:
            return response.status, await response.text()
