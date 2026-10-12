"""One localhost server standing in for every upstream a chat turn can call.

It answers in each upstream's wire format and keeps its own invoice: every
call it served, the turn it served it in, and what that call costs the
platform at the provider's prices. That invoice is what the platform's
charges are checked against.
"""

import uuid
from functools import partial
from typing import Literal

from aiohttp import web
from pydantic import BaseModel

from . import wire
from .invoice import CacheTtl, Call, Route, Usage, price_table, scripted_usage
from .replies import GREETING, anthropic_reply, cache_ttl, chat_reply, responses_reply


class PlannedFault(BaseModel):
    """A model request the provider fails, before doing any work it bills."""

    kind: Literal["overloaded", "dropped"]
    turn: int
    # Counts every model request of the turn, failed ones included, from 1.
    request: int


class FakeProvider:
    def __init__(self) -> None:
        self.turn = 0
        self.calls: list[Call] = []
        self.unexpected: list[str] = []
        self.generation_lookups_fail = False
        self.faults: list[PlannedFault] = []
        self.faults_served: list[PlannedFault] = []
        self.url = ""
        self._runner: web.AppRunner | None = None
        self._requests_per_turn: dict[int, int] = {}

    async def __aenter__(self) -> "FakeProvider":
        self._runner = web.AppRunner(self._app())
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def calls_in(self, turn: int) -> list[Call]:
        return [call for call in self.calls if call.turn == turn]

    def _app(self) -> web.Application:
        app = web.Application()
        upstreams: tuple[tuple[str, Route], ...] = (
            ("/anthropic", "anthropic"),
            ("/openrouter/api", "openrouter"),
            ("/ollama", "ollama"),
        )
        for prefix, route in upstreams:
            app.router.add_post(
                f"{prefix}/v1/chat/completions", partial(self._chat, route=route)
            )
            if route != "ollama":
                app.router.add_post(
                    f"{prefix}/v1/messages", partial(self._messages, route=route)
                )
        app.router.add_post("/codex/responses", self._responses)
        app.router.add_post("/graph/copilot/conversations", self._graph_conversation)
        app.router.add_post(
            "/graph/copilot/conversations/{id}/chatOverStream", self._graph_chat
        )
        app.router.add_get("/openrouter/api/v1/generation", self._generation)
        app.router.add_get("/ollama/api/ps", self._ollama_loaded_models)
        # The CLI checks it can reach its base URL before a session starts.
        app.router.add_route("HEAD", "/{prefix:.*}/api/hello", self._reachable)
        app.router.add_route("*", "/{tail:.*}", self._unexpected)
        return app

    async def _messages(self, request: web.Request, route: Route) -> web.StreamResponse:
        if failed := self._fault(request):
            return failed
        body = await request.json()
        ttl = cache_ttl(body)
        usage = scripted_usage(len(self.calls), cached=ttl is not None)
        generation_id = f"gen-{uuid.uuid4().hex}" if route == "openrouter" else None
        self._bill(route, body["model"], usage, ttl, generation_id)
        sse = wire.anthropic_messages(
            message_id=generation_id or f"msg_{uuid.uuid4().hex}",
            model=body["model"],
            reply=anthropic_reply(body["messages"]),
            usage=usage,
            ttl=ttl,
        )
        return await _stream(request, sse)

    async def _chat(self, request: web.Request, route: Route) -> web.StreamResponse:
        if failed := self._fault(request):
            return failed
        body = await request.json()
        ttl = cache_ttl(body)
        usage = scripted_usage(len(self.calls), cached=ttl is not None)
        call = self._bill(route, body["model"], usage, ttl, None)
        sse = wire.chat_completion_chunks(
            model=body["model"],
            reply=chat_reply(body["messages"]),
            usage=usage,
            cost=call.invoice_usd if route == "openrouter" else None,
            cache_write_field={
                "openrouter": "cache_write_tokens",
                "anthropic": "cache_creation_input_tokens",
            }.get(route),
        )
        return await _stream(request, sse)

    async def _responses(self, request: web.Request) -> web.StreamResponse:
        if failed := self._fault(request):
            return failed
        body = await request.json()
        scripted = scripted_usage(len(self.calls), cached=True)
        usage = Usage(
            input_tokens=scripted.input_tokens + scripted.cache_write_tokens,
            output_tokens=scripted.output_tokens,
            cache_read_tokens=scripted.cache_read_tokens,
        )
        self._bill("codex", body["model"], usage, None, None)
        sse = wire.responses_events(
            response_id=f"resp_{uuid.uuid4().hex}",
            model=body["model"],
            reply=responses_reply(body["input"]),
            usage=usage,
        )
        return await _stream(request, sse)

    async def _graph_conversation(self, _request: web.Request) -> web.Response:
        return web.json_response({"id": f"conv-{uuid.uuid4().hex[:12]}"}, status=201)

    async def _graph_chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body["message"]["text"]
        # Graph reports no usage; the platform estimates it from the text.
        self._bill(
            "graph",
            "microsoft-365-copilot",
            Usage(input_tokens=0, output_tokens=0),
            None,
            None,
        )
        sse = wire.graph_chat_snapshots(
            conversation_id=request.match_info["id"], prompt=prompt, answer=GREETING
        )
        return await _stream(request, sse)

    async def _generation(self, request: web.Request) -> web.Response:
        generation_id = request.query.get("id", "")
        call = next((c for c in self.calls if c.generation_id == generation_id), None)
        if self.generation_lookups_fail:
            # A permanent refusal, which the reconcile does not retry.
            return web.json_response({"error": {"code": 403}}, status=403)
        if call is None:
            self.unexpected.append(f"generation lookup for unknown {generation_id}")
            return web.json_response({"error": {"code": 404}}, status=404)
        return web.json_response(
            {
                "data": {
                    "id": generation_id,
                    "model": call.model,
                    "total_cost": call.invoice_usd,
                }
            }
        )

    async def _ollama_loaded_models(self, _request: web.Request) -> web.Response:
        # The local transport reads the loaded model's window before a turn.
        return web.json_response(
            {"models": [{"name": "llama3.1:8b", "context_length": 131_072}]}
        )

    async def _reachable(self, _request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _unexpected(self, request: web.Request) -> web.Response:
        self.unexpected.append(f"{request.method} {request.path}")
        return web.json_response({"error": {"message": "not served"}}, status=404)

    def _fault(self, request: web.Request) -> web.Response | None:
        """Fail this model request if the plan says so: a 529, or a connection
        that drops before any answer."""
        nth = self._requests_per_turn.get(self.turn, 0) + 1
        self._requests_per_turn[self.turn] = nth
        fault = next(
            (f for f in self.faults if (f.turn, f.request) == (self.turn, nth)), None
        )
        if fault is None:
            return None
        self.faults_served.append(fault)
        if fault.kind == "dropped" and request.transport is not None:
            request.transport.abort()
        return web.json_response(
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
            status=529,
        )

    def _bill(
        self,
        route: Route,
        model: str,
        usage: Usage,
        ttl: CacheTtl | None,
        generation_id: str | None,
    ) -> Call:
        table = price_table(route)
        if table is not None and model not in table:
            self.unexpected.append(f"{route} has no price for {model}")
            raise web.HTTPBadRequest(text=f"unknown model {model}")
        call = Call(
            turn=self.turn,
            request=self._requests_per_turn.get(self.turn, 0),
            route=route,
            model=model,
            usage=usage,
            cache_ttl=ttl,
            invoice_usd=table[model].of(usage, ttl) if table is not None else 0.0,
            generation_id=generation_id,
        )
        self.calls.append(call)
        return call


async def _stream(request: web.Request, body: str) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    await response.write(body.encode())
    await response.write_eof()
    return response
