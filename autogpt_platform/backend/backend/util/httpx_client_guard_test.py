"""Tests for the guard that fails a test leaving an httpx client open on its
closed event loop (``httpx_client_guard.py``)."""

import asyncio
from functools import partial

import httpx
import pytest

from .httpx_client_guard import HttpxClientGuard

_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"


@pytest.mark.parametrize(
    "transport, close_client, close_loop, named",
    [
        pytest.param("real", False, True, True, id="left-open-on-a-closed-loop"),
        pytest.param("real", True, True, False, id="closed-before-its-loop"),
        pytest.param("real", False, False, False, id="its-loop-still-runs"),
        pytest.param("mock", False, True, False, id="never-pooled-a-connection"),
    ],
)
def test_names_only_a_client_left_open_with_connections_on_a_closed_loop(
    transport: str, close_client: bool, close_loop: bool, named: bool
) -> None:
    guard = HttpxClientGuard()
    guard.start()
    loop = asyncio.new_event_loop()
    client = loop.run_until_complete(
        _request_once(guard, mock=transport == "mock", close=close_client)
    )
    if close_loop:
        loop.close()

    left_open = guard.stop()

    if not loop.is_closed():
        loop.run_until_complete(client.aclose())
        loop.close()
    assert len(left_open) == int(named)
    if named:
        assert "httpx_client_guard_test.py" in left_open[0]
        assert "127.0.0.1" in left_open[0]


def test_a_client_is_checked_only_in_the_test_that_sent_through_it() -> None:
    guard = HttpxClientGuard()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_request_once(guard, mock=False, close=False))
    loop.close()

    guard.start()
    assert guard.stop() == []


async def _request_once(
    guard: HttpxClientGuard, *, mock: bool, close: bool
) -> httpx.AsyncClient:
    """One request from a fresh client to a local server that keeps the
    connection alive, as a provider does, so the client pools it."""
    connections: list[asyncio.StreamWriter] = []
    server = await asyncio.start_server(partial(_answer, connections), "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    client = (
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        if mock
        else httpx.AsyncClient()
    )
    send = guard.wrap(_send_past_the_session_guard)
    response = await send(client, httpx.Request("GET", f"http://{host}:{port}/"))
    assert response.status_code == 200
    if close:
        await client.aclose()
    await _drop_sockets(client)
    for connection in connections:
        connection.close()
    server.close()
    await server.wait_closed()
    return client


async def _drop_sockets(client: httpx.AsyncClient) -> None:
    """Closes the socket under each pooled connection, which httpcore still
    lists as open, as it does when a server hangs up on an idle connection.
    The guard sees the same pool, and a client left open on purpose leaks no
    socket."""
    if not isinstance(client._transport, httpx.AsyncHTTPTransport):
        return
    for connection in client._transport._pool.connections:
        await connection._connection._network_stream.aclose()


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
