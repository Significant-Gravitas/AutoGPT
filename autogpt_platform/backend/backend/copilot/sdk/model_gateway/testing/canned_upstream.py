"""A loopback server that answers each request with the next canned reply.

For adapter tests that need one exact upstream behaviour -- an error body, a
stream cut short, a header -- and want to see exactly what was sent.
"""

import json
from typing import Any

from aiohttp import web
from pydantic import BaseModel, Field


class CannedReply(BaseModel):
    status: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes = b""
    # When set, the body is written in these pieces with the stream left open
    # between them, then closed without the rest.
    chunks: list[bytes] | None = None

    @classmethod
    def of_json(cls, payload: Any, status: int = 200, **headers: str) -> "CannedReply":
        return cls(
            status=status,
            headers={"Content-Type": "application/json", **headers},
            body=json.dumps(payload).encode(),
        )

    @classmethod
    def of_sse(
        cls, events: list[Any], *, named: bool = True, done: bool = False
    ) -> "CannedReply":
        frames = [
            (f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n"
            for e in events
        ]
        if done:
            frames.append("data: [DONE]\n\n")
        return cls(
            headers={"Content-Type": "text/event-stream"},
            body="".join(frames).encode(),
        )


class SeenCall(BaseModel):
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


class CannedUpstream:
    def __init__(self, *replies: CannedReply) -> None:
        self._replies = list(replies)
        self.calls: list[SeenCall] = []
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> "CannedUpstream":
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._answer)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _answer(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        self.calls.append(
            SeenCall(
                path=request.path_qs,
                headers={k.lower(): v for k, v in request.headers.items()},
                body=json.loads(raw) if raw else {},
            )
        )
        reply = self._replies.pop(0) if self._replies else CannedReply(status=599)
        if reply.chunks is None:
            return web.Response(
                status=reply.status, headers=reply.headers, body=reply.body
            )
        response = web.StreamResponse(status=reply.status, headers=reply.headers)
        await response.prepare(request)
        for chunk in reply.chunks:
            await response.write(chunk)
        if request.transport is not None:
            request.transport.close()
        return response
