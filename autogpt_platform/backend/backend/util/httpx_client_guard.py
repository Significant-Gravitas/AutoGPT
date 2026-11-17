"""Find LLM SDK clients a test left open on an event loop that has closed.

The OpenAI, Anthropic and Groq SDKs talk through an ``httpx.AsyncClient`` of
their own, whose ``__del__`` schedules ``aclose()`` on whatever event loop is
running when it is garbage-collected. A test on an event loop of its own
(``loop_scope="function"``, a ``TestClient``'s) that leaves one open with
pooled connections leaves them bound to a loop that is gone. Once collected,
the client's ``aclose()`` fails there with ``Event loop is closed``, reported
as "Task exception was never retrieved" in a later, unrelated test.

A plain ``httpx.AsyncClient`` has no such finalizer. A shared one (Prisma's
query-engine client, a service's RPC client) can hold connections from a
loop a test closed without passing a failure on this way, so it is not
checked.
"""

import asyncio
import os
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from anthropic._base_client import AsyncHttpxClientWrapper as AnthropicHttpxClient
from groq._base_client import AsyncHttpxClientWrapper as GroqHttpxClient
from openai._base_client import AsyncHttpxClientWrapper as OpenAIHttpxClient

# The SDK clients whose __del__ runs aclose() on the running loop.
CLOSED_WHEN_COLLECTED = (OpenAIHttpxClient, AnthropicHttpxClient, GroqHttpxClient)

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PACKAGE = os.path.join(_BACKEND_ROOT, "backend") + os.sep

Send = Callable[..., Awaitable[httpx.Response]]


class HttpxClientGuard:
    """Remembers each SDK client a test sends through and the loop it sent
    on, then names the ones left open there once that loop has closed."""

    def __init__(self) -> None:
        # Strong references, so a client the test dropped is still here to
        # check, however the garbage collector's timing falls.
        self._sent: dict[
            int, tuple[httpx.AsyncClient, asyncio.AbstractEventLoop, str]
        ] = {}
        self._watching = False

    def start(self) -> None:
        self._sent.clear()
        self._watching = True

    def stop(self) -> list[str]:
        """Each client left open with connections on a loop that has closed."""
        self._watching = False
        sent, self._sent = self._sent, {}
        return [
            described
            for client, loop, described in sent.values()
            if loop.is_closed() and not client.is_closed and _pooled(client)
        ]

    def wrap(self, send: Send) -> Send:
        """``send``, recording each SDK client on its first request of the
        test."""

        async def recorded_send(
            client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
        ) -> httpx.Response:
            if (
                self._watching
                and isinstance(client, CLOSED_WHEN_COLLECTED)
                and id(client) not in self._sent
            ):
                self._sent[id(client)] = (
                    client,
                    asyncio.get_running_loop(),
                    _describe(client, request),
                )
            return await send(client, request, **kwargs)

        return recorded_send


def left_open_report(nodeid: str, left_open: list[str]) -> str:
    """What a test that left clients open is told, and how to fix it."""
    return (
        f"{nodeid} left {len(left_open)} OpenAI/Anthropic/Groq client(s) open "
        "on an event loop that has closed:\n"
        + "\n".join(f"  {client}" for client in left_open)
        + "\nTheir pooled connections can no longer be closed. Once such a "
        "client is garbage-collected, it runs aclose() on the loop of whatever "
        'test is running, which raises "Event loop is closed" ("Task exception '
        'was never retrieved") there. Close each client before its loop ends: '
        "`async with` it, or close it in the fixture that made it."
    )


def _pooled(client: httpx.AsyncClient) -> int:
    """How many connections the client's pools still hold open."""
    transports = [client._transport, *client._mounts.values()]
    return sum(
        not connection.is_closed()
        for transport in transports
        if isinstance(transport, httpx.AsyncHTTPTransport)
        for connection in transport._pool.connections
    )


def _describe(client: httpx.AsyncClient, request: httpx.Request) -> str:
    origin = f"{request.url.scheme}://{request.url.netloc.decode()}"
    sdk = type(client).__module__.split(".")[0]
    return f"{sdk} client to {origin}, first used at {_caller()}"


def _caller() -> str:
    """The innermost backend frame outside this module."""
    for frame, line in traceback.walk_stack(None):
        path = frame.f_code.co_filename
        if path.startswith(_PACKAGE) and path != __file__:
            return f"{os.path.relpath(path, _BACKEND_ROOT)}:{line} in {frame.f_code.co_name}"
    return "a caller outside the backend"
