"""One loopback server playing the scripted model behind three wire protocols.

- ``/anthropic/v1/messages``: what the pass-through forwards to;
- ``/openai/v1/chat/completions``: the Chat Completions adapter's upstream;
- ``/codex/responses``: the Codex adapter's, through an HTTPX client
  pointed at ``{url}/codex``.

Every request is recorded with the decision it got, so a test can check what
the gateway actually sent upstream.
"""

import json
from typing import Any, Literal

from aiohttp import web
from pydantic import BaseModel

from backend.copilot.sdk.model_gateway.testing.script import Decision, Script
from backend.copilot.sdk.model_gateway.testing.wire import (
    LLAMA_CPP_OVERFLOW,
    VLLM_OVERFLOW,
    anthropic_events,
    anthropic_message,
    chat_chunks,
    responses_events,
    sse,
)

Protocol = Literal["anthropic", "chat", "responses"]


class SeenRequest(BaseModel):
    protocol: Protocol
    body: dict[str, Any]
    decision: Decision


class ScriptedUpstream:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.seen: list[SeenRequest] = []
        self.unexpected: list[str] = []
        self.issued_reasoning: list[str] = []
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> "ScriptedUpstream":
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/anthropic/v1/messages", self._anthropic)
        app.router.add_post("/openai/v1/chat/completions", self._chat)
        app.router.add_post("/codex/responses", self._responses)
        app.router.add_route("*", "/{tail:.*}", self._unexpected)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    @property
    def compactions(self) -> list[SeenRequest]:
        return [seen for seen in self.seen if seen.decision.compaction]

    async def _anthropic(self, request: web.Request) -> web.StreamResponse:
        body, size = await _read(request)
        messages = body.get("messages") or []
        decision = self._decide("anthropic", body, _anthropic_latest(messages), size)
        if decision.kind == "overflow":
            return web.json_response(LLAMA_CPP_OVERFLOW, status=400)
        message = anthropic_message(
            decision,
            model=str(body.get("model")),
            tool=_tool_name([t.get("name") for t in body.get("tools") or []]),
            n=len(self.seen),
            input_tokens=self.script.input_tokens(size),
        )
        if body.get("stream") is True:
            return await _stream(request, sse(anthropic_events(message)))
        return web.json_response(message)

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body, size = await _read(request)
        decision = self._decide(
            "chat", body, _chat_latest(body.get("messages") or []), size
        )
        if decision.kind == "overflow":
            return web.json_response(VLLM_OVERFLOW, status=400)
        tools = [(t.get("function") or {}).get("name") for t in body.get("tools") or []]
        chunks = chat_chunks(
            decision,
            model=str(body.get("model")),
            tool=_tool_name(tools),
            n=len(self.seen),
            prompt_tokens=self.script.input_tokens(size),
            fully_cached=self.script.fully_cached,
        )
        return await _stream(request, sse(chunks, named=False) + b"data: [DONE]\n\n")

    async def _responses(self, request: web.Request) -> web.StreamResponse:
        body, size = await _read(request)
        decision = self._decide(
            "responses", body, _responses_latest(body.get("input") or []), size
        )
        encrypted = f"gAAAAAB-stub-reasoning-{len(self.seen)}-" + "Q" * 4096
        if decision.kind != "overflow":
            self.issued_reasoning.append(encrypted)
        events = responses_events(
            decision,
            model=str(body.get("model")),
            tool=_tool_name([t.get("name") for t in body.get("tools") or []]),
            n=len(self.seen),
            input_tokens=self.script.input_tokens(size),
            encrypted=encrypted,
            summarize=self.script.reasoning_summary,
            fully_cached=self.script.fully_cached,
        )
        return await _stream(request, sse(events))

    async def _unexpected(self, request: web.Request) -> web.Response:
        self.unexpected.append(f"{request.method} {request.path}")
        return web.json_response({"error": {"message": "not served"}}, status=404)

    def _decide(
        self, protocol: Protocol, body: dict[str, Any], latest: str, size: int
    ) -> Decision:
        everything = json.dumps(body, ensure_ascii=False)
        decision = self.script.decide(latest, everything, size)
        self.seen.append(SeenRequest(protocol=protocol, body=body, decision=decision))
        return decision


async def _read(request: web.Request) -> tuple[dict[str, Any], int]:
    raw = await request.read()
    body = json.loads(raw)
    return (body if isinstance(body, dict) else {}), len(raw)


async def _stream(request: web.Request, payload: bytes) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    await response.write(payload)
    await response.write_eof()
    return response


def _tool_name(names: list[Any]) -> str:
    return next(
        (n for n in names if isinstance(n, str) and n.endswith("read_doc")), "read_doc"
    )


def _anthropic_latest(messages: list[Any]) -> str:
    """Text of everything after the last assistant message."""
    tail = _after_last(messages, lambda m: m.get("role") == "assistant")
    return "\n".join(_texts(m.get("content")) for m in tail)


def _chat_latest(messages: list[Any]) -> str:
    tail = _after_last(messages, lambda m: m.get("role") == "assistant")
    return "\n".join(_texts(m.get("content")) for m in tail)


def _responses_latest(items: list[Any]) -> str:
    tail = _after_last(
        items,
        lambda i: i.get("role") == "assistant"
        or i.get("type") in ("function_call", "reasoning"),
    )
    return "\n".join(_texts(i.get("content") or i.get("output")) for i in tail)


def _after_last(items: list[Any], is_model_turn: Any) -> list[dict[str, Any]]:
    dicts = [item for item in items if isinstance(item, dict)]
    last = max((i for i, item in enumerate(dicts) if is_model_turn(item)), default=-1)
    return dicts[last + 1 :]


def _texts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict):
            inner = block.get("content")
            parts.append(
                str(block.get("text") or "") + (_texts(inner) if inner else "")
            )
    return "\n".join(parts)
