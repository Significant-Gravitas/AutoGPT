"""Local servers for the ``Requests`` tests, and the clients that reach them."""

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TypeVar
from unittest.mock import patch

import pytest
from aiohttp import web
from pydantic import BaseModel

from backend.util import request as request_module
from backend.util.request import Requests

# Every request in these tests talks to a local server, so its attempts take
# milliseconds; the bound only trips when the retries never stop.
ENDS_WITHIN_SECONDS = 10

# The two clients that retry a throttle status: one raising its errors, with a
# cap, and one checking statuses itself, on the default.
RETRYING_CLIENTS = pytest.mark.parametrize(
    "client_kwargs",
    [{"retry_max_attempts": 3}, {"raise_for_status": False}],
    ids=["raise_for_status-with-a-cap", "status-checked-by-the-caller"],
)


async def serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


def local_requests(**kwargs) -> Requests:
    return Requests(trusted_origins=["127.0.0.1"], **kwargs)


# Far more waits than any cap in these tests asks for.
MOST_RETRY_WAITS = 10_000


async def retry_waits(client: Requests) -> list[float]:
    """The waits between the attempts of one of *client*'s requests that meets
    a throttle status every time: what its retries would sleep, noted instead.

    Fails at ``MOST_RETRY_WAITS``: noting a wait never yields to the event
    loop, so a budget that never runs out would otherwise hang the test."""
    waits: list[float] = []

    async def note(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) >= MOST_RETRY_WAITS:
            raise AssertionError(
                f"The retry budget never ran out: {len(waits)} waits, so it "
                "is unbounded."
            )

    with patch.object(request_module, "sleep", note):
        budget = client._retry_budget()
        while await budget.send_again():
            pass
    return waits


def scripted(*statuses: int) -> tuple[web.Application, list[int]]:
    """An app answering *statuses* in turn, repeating the last for good, and
    the statuses it has served. Each body names its attempt."""
    served: list[int] = []

    async def answer(request: web.Request) -> web.Response:
        status = statuses[min(len(served), len(statuses) - 1)]
        served.append(status)
        return web.Response(status=status, text=f"attempt {len(served)}")

    app = web.Application()
    app.router.add_route("*", "/", answer)
    return app, served


# --- Two origins that note what each hop of a redirect received ---


class Received(BaseModel):
    """One request as a test server received it, with its header names
    lower-cased. A multipart body's boundary, new on every send, reads
    ``BOUNDARY`` in the body and the Content-Type alike."""

    server: str
    method: str
    path: str
    query: str
    content_type: str | None
    body: bytes
    headers: dict[str, str]


Answer = Callable[[web.Request], web.StreamResponse]


class Origin:
    """A local server answering ``routes`` by path, which notes each request
    it receives in a list it shares with the other origins of a test."""

    def __init__(self, name: str, received: list[Received]):
        self.name = name
        self.routes: dict[str, Answer] = {}
        self.base = ""
        self._received = received

    def app(self) -> web.Application:
        async def handle(request: web.Request) -> web.StreamResponse:
            self._received.append(_received(self.name, request, await request.read()))
            answer = self.routes.get(request.path)
            return answer(request) if answer else web.Response(status=404)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        return app


@asynccontextmanager
async def origins() -> AsyncIterator[tuple[Origin, Origin, list[Received]]]:
    """Two local servers on two ports, so two origins, and what they receive.
    Each is cleaned up however the other's start or cleanup goes."""
    received: list[Received] = []
    first, second = Origin("first", received), Origin("second", received)
    async with AsyncExitStack() as stack:
        for origin in (first, second):
            runner, origin.base = await serve(origin.app())
            stack.push_async_callback(runner.cleanup)
        yield first, second, received


def _received(server: str, request: web.Request, body: bytes) -> Received:
    content_type = request.headers.get("Content-Type")
    boundary = re.search(r"boundary=([^;\s]+)", content_type or "")
    if content_type is not None and boundary:
        content_type = content_type.replace(boundary.group(1), "BOUNDARY")
        body = body.replace(boundary.group(1).encode(), b"BOUNDARY")
    return Received(
        server=server,
        method=request.method,
        path=request.path,
        query=request.query_string,
        content_type=content_type,
        body=body,
        headers={name.lower(): value for name, value in request.headers.items()},
    )


def redirect(status: int, location: str) -> Answer:
    return lambda request: web.Response(status=status, headers={"Location": location})


def done(request: web.Request) -> web.Response:
    return web.Response(text="done")


def answers(*steps: Answer) -> Answer:
    """Each step in turn, then the last one for good."""
    sent = 0

    def answer(request: web.Request) -> web.StreamResponse:
        nonlocal sent
        sent += 1
        return steps[min(sent, len(steps)) - 1](request)

    return answer


def status(code: int) -> Answer:
    return lambda request: web.Response(status=code, text=f"status {code}")


def location(target: Origin, origin: str) -> str:
    """Where a redirect from the first origin sends the next hop."""
    return "/next" if origin == "same-origin" else f"{target.base}/next"


_Result = TypeVar("_Result")


async def bounded(sent: Awaitable[_Result]) -> _Result:
    return await asyncio.wait_for(sent, ENDS_WITHIN_SECONDS)


def hops(received: list[Received]) -> list[tuple[str, str, str]]:
    return [(hop.server, hop.method, hop.path) for hop in received]


def without_body(hop: Received) -> bool:
    return (
        hop.body == b""
        and hop.content_type is None
        and "content-length" not in hop.headers
        and "transfer-encoding" not in hop.headers
    )
