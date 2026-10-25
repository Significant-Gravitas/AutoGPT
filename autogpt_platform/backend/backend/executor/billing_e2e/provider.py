"""A localhost Anthropic Messages endpoint that plays the model for an agent run.

It answers the Orchestrator's calls (they offer tools) by calling the first
tool it was offered a set number of times, then finishing, and it answers the
AI block the agent calls as a tool (its calls offer none) with text, or with
an error when the run is scripted to fail its tool calls. It keeps an invoice
of every reply, with the time each call came in, and each billed reply reports
one million input tokens, so it costs the platform exactly the model's price
per million input tokens.
"""

import time
from typing import Literal

from aiohttp import web
from pydantic import BaseModel, Field

INPUT_TOKENS_PER_REPLY = 1_000_000


class Reply(BaseModel):
    caller: Literal["agent", "tool"]
    billed: bool
    at: float = Field(default_factory=time.monotonic)


class FakeAnthropic:
    def __init__(self, *, tool_calls: int, tool_calls_fail: bool) -> None:
        self.tool_calls = tool_calls
        self.tool_calls_fail = tool_calls_fail
        self.replies: list[Reply] = []
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> "FakeAnthropic":
        app = web.Application()
        app.router.add_post("/v1/messages", self._messages)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._runner:
            await self._runner.cleanup()

    def calls(self, caller: Literal["agent", "tool"]) -> int:
        return sum(r.caller == caller for r in self.replies)

    def tool_call_delays(self) -> list[float]:
        """Seconds from each agent reply to the tool call that came next."""
        return [
            tool.at - agent.at
            for agent, tool in zip(self.replies, self.replies[1:])
            if agent.caller == "agent" and tool.caller == "tool"
        ]

    def billed(self) -> int:
        """How many replies the provider billed the platform for."""
        return sum(r.billed for r in self.replies)

    async def _messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        if body.get("tools"):
            return self._agent_turn(body)
        if self.tool_calls_fail:
            self.replies.append(Reply(caller="tool", billed=False))
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "The tool's model call failed.",
                    },
                },
                status=400,
            )
        self.replies.append(Reply(caller="tool", billed=True))
        return _message(body, [{"type": "text", "text": "Hello!"}], "end_turn")

    def _agent_turn(self, body: dict) -> web.Response:
        self.replies.append(Reply(caller="agent", billed=True))
        # Requests carry the whole conversation: each earlier turn of the
        # agent is one assistant message in it.
        turn = sum(m["role"] == "assistant" for m in body["messages"])
        if turn >= self.tool_calls:
            return _message(body, [{"type": "text", "text": "Done."}], "end_turn")
        tool = body["tools"][0]
        call = {
            "type": "tool_use",
            "id": f"toolu_{turn}",
            "name": tool["name"],
            "input": {
                field: "Say hello." for field in tool["input_schema"]["properties"]
            },
        }
        return _message(body, [call], "tool_use")


def _message(body: dict, content: list[dict], stop_reason: str) -> web.Response:
    return web.json_response(
        {
            "id": "msg_billing_e2e",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": INPUT_TOKENS_PER_REPLY,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )
