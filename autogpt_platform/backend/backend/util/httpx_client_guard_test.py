"""Tests for the guard that fails a test leaving an LLM SDK client open on its
closed event loop (``httpx_client_guard.py``)."""

import asyncio
from functools import partial

import httpx
import pytest
from openai._base_client import AsyncHttpxClientWrapper

from .httpx_client_guard import HttpxClientGuard

_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"


@pytest.mark.parametrize(
    "client_kind, close_client, close_loop, named",
    [
        pytest.param("sdk", False, True, True, id="left-open-on-a-closed-loop"),
        pytest.param("sdk", True, True, False, id="closed-before-its-loop"),
        pytest.param("sdk", False, False, False, id="its-loop-still-runs"),
        pytest.param("sdk-mock", False, True, False, id="never-pooled-a-connection"),
        pytest.param("plain", False, True, False, id="no-finalizer-to-run-aclose"),
    ],
)
def test_names_only_an_sdk_client_left_open_with_connections_on_a_closed_loop(
    client_kind: str, close_client: bool, close_loop: bool, named: bool
) -> None:
    guard = HttpxClientGuard()
    guard.start()
    client = _client(client_kind)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_request_once(guard, client, close=close_client))
        if close_loop:
            loop.close()

        left_open = guard.stop()
    finally:
        _close(client, loop)

    assert len(left_open) == int(named)
    if named:
        assert left_open[0].startswith("openai client to http://127.0.0.1:")
        assert "httpx_client_guard_test.py" in left_open[0]


def test_a_client_is_checked_only_in_the_test_that_sent_through_it() -> None:
    guard = HttpxClientGuard()
    client = _client("sdk")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_request_once(guard, client, close=False))
        loop.close()

        guard.start()
        left_open = guard.stop()
    finally:
        _close(client, loop)

    assert left_open == []


def _client(kind: str) -> httpx.AsyncClient:
    if kind == "sdk-mock":
        return AsyncHttpxClientWrapper(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        )
    return AsyncHttpxClientWrapper() if kind == "sdk" else httpx.AsyncClient()


async def _request_once(
    guard: HttpxClientGuard, client: httpx.AsyncClient, *, close: bool
) -> None:
    """One request from ``client`` to a local server that keeps the
    connection alive, as a provider does, so the client pools it. The server
    and the client's sockets are released however the request goes."""
    connections: list[asyncio.StreamWriter] = []
    server = await asyncio.start_server(partial(_answer, connections), "127.0.0.1", 0)
    try:
        host, port = server.sockets[0].getsockname()[:2]
        send = guard.wrap(_send_past_the_session_guard)
        response = await send(client, httpx.Request("GET", f"http://{host}:{port}/"))
        assert response.status_code == 200
        if close:
            await client.aclose()
    finally:
        await _drop_sockets(client)
        for connection in connections:
            connection.close()
        server.close()
        await server.wait_closed()


async def _drop_sockets(client: httpx.AsyncClient) -> None:
    """Closes the socket under each pooled connection, which httpcore still
    lists as open, as it does when a server hangs up on an idle connection.
    The guard sees the same pool, and a client left open on purpose leaks no
    socket."""
    if not isinstance(client._transport, httpx.AsyncHTTPTransport):
        return
    for connection in client._transport._pool.connections:
        await connection._connection._network_stream.aclose()


def _close(client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop) -> None:
    """Closes a client left open on purpose, so its finalizer has nothing to
    do, and then the loop. Its sockets are already gone, so any loop can
    close it."""
    if loop.is_closed():
        loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(client.aclose())
    finally:
        loop.close()


async def _send_past_the_session_guard(
    client: httpx.AsyncClient, request: httpx.Request
) -> httpx.Response:
    """Sends through the client's transport, not ``AsyncClient.send``, so the
    session's own guard doesn't fail this test for the client it leaves open
    on purpose."""
    response = await client._transport.handle_async_request(request)
    await response.aread()
    await response.aclose()
    return response


async def _answer(
    connections: list[asyncio.StreamWriter],
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(_RESPONSE)
    await writer.drain()
    connections.append(writer)
