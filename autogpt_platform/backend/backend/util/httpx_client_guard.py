"""Find httpx clients a test left open on an event loop that has closed.

A test on an event loop of its own (``loop_scope="function"``, a
``TestClient``'s) that leaves an ``httpx.AsyncClient`` open with pooled
connections leaves them bound to a loop that is gone: nothing can close them
any more. The OpenAI and Anthropic SDKs pass the damage on. When their client
is garbage-collected, its ``__del__`` schedules ``aclose()`` on whatever loop
is running then, where it fails with ``Event loop is closed`` and is reported
as "Task exception was never retrieved", in a later, unrelated test.
"""

import asyncio
import os
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PACKAGE = os.path.join(_BACKEND_ROOT, "backend") + os.sep

Send = Callable[..., Awaitable[httpx.Response]]


class HttpxClientGuard:
    """Remembers each client a test sends through and the loop it sent on,
    then names the ones left open there once that loop has closed."""

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
        """``send``, recording each client on its first request of the test."""

        async def recorded_send(
            client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
        ) -> httpx.Response:
            if self._watching and id(client) not in self._sent:
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
        f"{nodeid} left {len(left_open)} httpx client(s) open on an event loop "
        "that has closed:\n"
        + "\n".join(f"  {client}" for client in left_open)
        + "\nTheir pooled connections can no longer be closed, and an OpenAI or "
        "Anthropic client then fails a later test: garbage-collected, it runs "
        'aclose() on that test\'s loop, which raises "Event loop is closed" '
        '("Task exception was never retrieved"). Close each client before its '
        "loop ends: `async with` it, or close it in the fixture that made it."
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
    return f"{type(client).__name__} to {origin}, first used at {_caller()}"


def _caller() -> str:
    """The innermost backend frame outside this module."""
    for frame, line in traceback.walk_stack(None):
        path = frame.f_code.co_filename
        if path.startswith(_PACKAGE) and path != __file__:
            return f"{os.path.relpath(path, _BACKEND_ROOT)}:{line} in {frame.f_code.co_name}"
    return "a caller outside the backend"
