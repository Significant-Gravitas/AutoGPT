"""Local Anthropic and OpenRouter HTTP responses for dream billing tests."""

from collections.abc import AsyncIterator

import anthropic
import openai
import pytest
import pytest_asyncio
from aiohttp import web

from backend.copilot.inference import routing as inference_routing
from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.util.llm import providers


class DreamProvider:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.stop_reason = "tool_use"
        self.prose = False
        self.cost: float | None = None
        self.usage = {
            "input_tokens": 1001,
            "output_tokens": 203,
            "cache_read_input_tokens": 3007,
            "cache_creation_input_tokens": 509,
        }

    async def messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        tool = body["tools"][0]
        content = [
            {"type": "text", "text": "Here is the result."},
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": tool["name"],
                "input": {"facts": []},
            },
        ]
        if self.prose:
            content = [
                {"type": "text", "text": 'Here it is:\n```json\n{"facts": []}\n```'}
            ]
        return web.json_response(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": content,
                "stop_reason": self.stop_reason,
                "stop_sequence": None,
                "usage": self.usage,
            }
        )

    async def chat(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        return web.json_response(
            {
                "id": "gen-test",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"facts": []}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 4517,
                    "completion_tokens": 203,
                    "total_tokens": 4720,
                    "prompt_tokens_details": {
                        "cached_tokens": 3007,
                        "cache_write_tokens": 509,
                    },
                    "cost": self.cost,
                },
            }
        )


@pytest_asyncio.fixture(loop_scope="function")
async def dream_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[DreamProvider]:
    provider = DreamProvider()
    app = web.Application()
    app.router.add_post("/v1/messages", provider.messages)
    app.router.add_post("/v1/chat/completions", provider.chat)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    host, port = runner.addresses[0][:2]
    async with anthropic.AsyncAnthropic(
        api_key="sk-test", base_url=f"http://{host}:{port}", max_retries=0
    ) as client, openai.AsyncOpenAI(
        api_key="sk-test", base_url=f"http://{host}:{port}/v1", max_retries=0
    ) as openai_client:
        monkeypatch.setattr(providers.openai, "AsyncOpenAI", lambda **_: openai_client)
        monkeypatch.setattr(providers.anthropic, "AsyncAnthropic", lambda **_: client)
        routing = ProviderRoutingKwargs(
            provider="anthropic",
            api_key="sk-test",
            base_url=None,
            supports_flex=False,
            cost_log_provider="anthropic",
        )
        monkeypatch.setattr(
            inference_routing, "routing_kwargs_for_chat_transport", lambda: routing
        )
        try:
            yield provider
        finally:
            await runner.cleanup()
