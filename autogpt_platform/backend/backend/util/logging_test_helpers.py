"""Serve a service's app the way its entrypoint does, and capture what Sentry
would be sent, for tests of what a request leaves behind. The fixtures built on
these are in this directory's conftest.py."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import uvicorn
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from backend.util.process import AppProcess


@asynccontextmanager
async def serve(
    process: AppProcess, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[str]:
    """Serve what ``process.run()`` hands uvicorn, on a free local port and
    without the lifespan's database and Redis connections."""
    runs: list[tuple] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: runs.append((app, kwargs))
    )
    # AgentServer.run() turns asyncio debug on for local runs; undo it after.
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "")
    process.run()
    [(app, kwargs)] = runs
    server = uvicorn.Server(
        uvicorn.Config(
            app, **{**kwargs, "host": "127.0.0.1", "port": 0, "lifespan": "off"}
        )
    )
    serving = asyncio.create_task(server.serve())
    for _ in range(1000):
        if server.started or serving.done():
            break
        await asyncio.sleep(0.01)
    assert server.started, "uvicorn did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await serving


class CapturingTransport(Transport):
    """Keeps every envelope the SDK sends: ``payload`` as sent, and the
    ``events`` in it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self.payload = ""

    def capture_envelope(self, envelope: Envelope) -> None:
        self.payload += envelope.serialize().decode()
        self.events += [item.payload.json or {} for item in envelope.items]
