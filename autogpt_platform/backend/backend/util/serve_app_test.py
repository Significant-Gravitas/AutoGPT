"""Serving an ASGI app under uvicorn for a test: on a free local port, shut
down within a bound, and leaving no request running after the test.

``serve`` runs what an ``AppProcess`` hands uvicorn; ``serve_app`` serves an
app directly. Both are used by the logging and scrubbing tests that need a real
server.
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from starlette.types import ASGIApp

from backend.api import ws_api
from backend.monitoring.instrumentation import METRICS_ENV_VAR
from backend.util.process import AppProcess


async def test_serve_cancels_a_server_whose_startup_never_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """uvicorn reads ``should_exit`` only once it has started, so a server
    stuck starting up is cancelled on the way out, not awaited forever."""
    stalled, released, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def startup(self: uvicorn.Server, sockets: object = None) -> None:
        stalled.set()
        try:
            await released.wait()
        finally:
            exited.set()

    monkeypatch.setattr(uvicorn.Server, "startup", startup)

    async def enter() -> None:
        async with serve(ws_api.WebsocketServer(), monkeypatch):
            pass

    entering = asyncio.create_task(enter())
    stalling = asyncio.create_task(stalled.wait())
    try:
        # Each wait is far longer than what it waits for; only a hang ends it.
        done, _ = await asyncio.wait(
            {entering, stalling}, timeout=30, return_when=asyncio.FIRST_COMPLETED
        )
        if entering in done:
            entering.result()  # whatever failed before uvicorn started up
        assert stalling in done, "uvicorn never started up"
        entering.cancel()
        done, _ = await asyncio.wait({entering}, timeout=30)
        assert entering in done and entering.cancelled()
        assert exited.is_set()
    finally:
        # A startup the cleanup left waiting ends here, so no task outlives
        # the test.
        released.set()
        stalling.cancel()
        _, pending = await asyncio.wait({entering, stalling}, timeout=30)
        for task in pending:
            task.cancel()


async def test_serve_fails_when_a_requests_cleanup_outlasts_its_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request still cleaning up after the wait is cancelled again, so it
    can't outlive the test, and the test fails."""
    monkeypatch.setattr(sys.modules[__name__], "_CLEANUP_SECONDS", 0.2)
    started, ended = asyncio.Event(), asyncio.Event()

    async def app(scope: dict, receive: object, send: object) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            try:
                await asyncio.sleep(30)  # cleanup far longer than the wait
            finally:
                ended.set()

    with pytest.raises(AssertionError, match="still cleaning up"):
        async with (
            serve_app(app, log_config=None, timeout_graceful_shutdown=0.2) as host,
            httpx.AsyncClient(base_url=f"http://{host}") as client,
        ):
            request = asyncio.create_task(client.get("/hang"))
            starting = asyncio.create_task(started.wait())
            try:
                done, _ = await asyncio.wait(
                    {request, starting}, return_when=asyncio.FIRST_COMPLETED
                )
                if request in done:
                    request.result()  # it failed before the app ran
            finally:
                for task in (request, starting):
                    task.cancel()
                await asyncio.wait({request, starting})

    assert ended.is_set()


async def test_serve_ends_a_request_still_running_at_shutdown() -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def app(scope: dict, receive: object, send: object) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.2)  # cleanup that itself takes a while
            cleaned.set()

    async def run() -> None:
        async with (
            serve_app(app, log_config=None) as host,
            httpx.AsyncClient(base_url=f"http://{host}") as client,
        ):
            request = asyncio.create_task(client.get("/hang"))
            starting = asyncio.create_task(started.wait())
            try:
                done, _ = await asyncio.wait(
                    {request, starting}, return_when=asyncio.FIRST_COMPLETED
                )
                if request in done:
                    request.result()  # it failed before the app ran
            finally:
                for task in (request, starting):
                    task.cancel()
                await asyncio.wait({request, starting})

    running = asyncio.create_task(run())
    try:
        # Far longer than the shutdown deadline; only a shutdown that waits on
        # the request for good reaches it.
        done, _ = await asyncio.wait({running}, timeout=30)
        assert running in done
        running.result()
        assert cleaned.is_set()
    finally:
        running.cancel()
        await asyncio.wait({running}, timeout=30)


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
    # Nor a metrics listener of its own, which would outlive the test.
    monkeypatch.setenv(METRICS_ENV_VAR, "false")
    process.run()
    [(app, kwargs)] = runs
    async with serve_app(app, **kwargs) as host:
        yield host


@asynccontextmanager
async def serve_app(app: ASGIApp, **options: object) -> AsyncIterator[str]:
    """Serve ``app`` under uvicorn with ``options``, on a free local port and
    with no lifespan. A request still running at shutdown is cancelled after a
    few seconds, which uvicorn logs, rather than hold the test up."""
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            **{
                "timeout_graceful_shutdown": 5,
                **options,
                "host": "127.0.0.1",
                "port": 0,
                "lifespan": "off",
            },
        )
    )
    serving = asyncio.create_task(server.serve())
    try:
        await _wait_for_start(server, serving)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"127.0.0.1:{port}"
    finally:
        await _shut_down(server, serving)


async def _wait_for_start(server: uvicorn.Server, serving: asyncio.Task[None]) -> None:
    """Wait, a bounded while, for uvicorn to listen."""
    for _ in range(1000):
        if server.started or serving.done():
            break
        await asyncio.sleep(0.01)
    assert server.started, "uvicorn did not start"


async def _shut_down(server: uvicorn.Server, serving: asyncio.Task[None]) -> None:
    """Stop the server, and join the requests it cancelled, whether the test
    ended or was cancelled, so neither outlives the test."""
    if server.started:
        server.should_exit = True
    else:
        # uvicorn reads should_exit only once it has started.
        serving.cancel()
    try:
        await asyncio.wait({serving})
    except asyncio.CancelledError:
        # Cancelled while it shuts down: a server that has started finishes its
        # bounded shutdown, closing its listener, and one still starting is
        # cancelled, so neither is left running.
        if not server.started:
            serving.cancel()
        await asyncio.wait({serving})
        await _join(server.server_state.tasks)
        raise
    late = await _join(server.server_state.tasks)
    assert not late, f"{late} request(s) still cleaning up after shutdown"
    if not serving.cancelled():
        serving.result()


# How long a request uvicorn cancelled at its shutdown deadline gets to clean up.
_CLEANUP_SECONDS = 5


async def _join(tasks: set[asyncio.Task]) -> int:
    """Wait for the requests uvicorn cancelled at its shutdown deadline, which
    it doesn't wait for, so their cleanup ends with the test. One still
    cleaning up after ``_CLEANUP_SECONDS`` is cancelled again and drained;
    returns how many were."""
    if not (pending := [task for task in tasks if not task.done()]):
        return 0
    _, late = await asyncio.wait(pending, timeout=_CLEANUP_SECONDS)
    for task in late:
        task.cancel()
    if late:
        await asyncio.wait(late, timeout=_CLEANUP_SECONDS)
    return len(late)
