import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from aiohttp import web

from backend.copilot.sdk.model_gateway.adapters.anthropic_passthrough import (
    AnthropicPassthrough,
)
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.adapters.openai_chat import OpenAIChat
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.route import AdapterKind, GatewayRoute
from backend.copilot.sdk.model_gateway.stream import Event
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.codex_lease import codex_route
from backend.copilot.sdk.model_gateway.testing.wire import anthropic_events


class ScriptedByteStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, ending: Literal["eof", "reset", "open"]) -> None:
        self.body = body
        self.ending = ending
        self.read_after_body = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body
        self.read_after_body = True
        if self.ending == "reset":
            raise httpx.ReadError("fake connection reset")
        if self.ending == "open":
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def native_message(text: str = "answer") -> Event:
    return {
        "id": "msg_native",
        "model": "gateway-test-model",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 9},
    }


def native_events(text: str = "answer") -> list[Event]:
    return anthropic_events(
        {**native_message(text), "usage": {"input_tokens": 100, "output_tokens": 0}}
    )


class NativeRetryUpstream(CannedUpstream):
    def __init__(self, partial: CannedReply | None = None) -> None:
        super().__init__(partial or CannedReply.of_sse(native_events()[:2]))

    async def _answer(self, request: web.Request) -> web.StreamResponse:
        if self.calls:
            body = await request.json()
            reply = (
                CannedReply.of_sse(native_events("retry succeeded"))
                if body.get("stream")
                else CannedReply.of_json(native_message("retry succeeded"))
            )
            self._replies.append(reply)
        return await super()._answer(request)


@asynccontextmanager
async def streamed_gateway(
    adapter: AdapterKind, wire: ScriptedByteStream
) -> AsyncIterator[ModelGateway]:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=wire)),
        trust_env=False,
    ) as client:
        if adapter == "codex_responses":
            route = codex_route("gateway-test-model")
            implementation = CodexResponses(route, client=client)
        else:
            route = GatewayRoute(
                adapter=adapter,
                model="gateway-test-model",
                base_url="http://127.0.0.1:9/v1",
            )
            implementation = (
                AnthropicPassthrough(route, client=client)
                if adapter == "anthropic_passthrough"
                else OpenAIChat(route, client=client)
            )
        async with ModelGateway(route, upstream=implementation) as gateway:
            yield gateway
