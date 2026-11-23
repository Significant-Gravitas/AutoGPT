"""The REST API keeps a connection open between requests for longer than the
clients and proxies in front of it do.

Google's load balancer keeps an idle connection to the API open for up to
600 s, and a client keeps its connection for as long as it has requests to
send. uvicorn closes an idle connection after 5 s by default, so a request sent
on one just as uvicorn closes it gets no answer: a client sees a reset or
"socket hang up", and the load balancer answers 502.
"""

import asyncio
import contextlib
import inspect

import uvicorn
from starlette.types import Receive, Scope, Send

from backend.api.utils.uvicorn_options import rest_api_uvicorn_options
from backend.util.serve_app_test import serve_app
from backend.util.settings import Config

# Longer than uvicorn keeps an idle connection without the API's setting.
_DEFAULT_KEEP_ALIVE = Config.model_fields["server_keep_alive_timeout_seconds"].default

_IDLE_SECONDS = (
    inspect.signature(uvicorn.Config).parameters["timeout_keep_alive"].default + 1
)


async def test_a_connection_left_idle_past_uvicorns_default_still_answers():
    # The default, not whatever backend/.env or the environment sets.
    config = Config(server_keep_alive_timeout_seconds=_DEFAULT_KEEP_ALIVE)
    async with serve_app(_ok, **rest_api_uvicorn_options(config)) as host:
        address, port = host.rsplit(":", 1)
        reader, writer = await asyncio.open_connection(address, int(port))
        try:
            assert await _get(reader, writer) == 200
            await asyncio.sleep(_IDLE_SECONDS)
            assert await _get(reader, writer) == 200, (
                f"the server closed the connection after it sat idle for "
                f"{_IDLE_SECONDS}s, so the request sent on it got no answer"
            )
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


async def _ok(scope: Scope, receive: Receive, send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", b"0")],
        }
    )
    await send({"type": "http.response.body", "body": b""})


async def _get(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> int | None:
    """Send `GET /` on the open connection: the answer's status, or None when
    the server closed the connection instead of answering."""
    writer.write(b"GET / HTTP/1.1\r\nHost: api\r\n\r\n")
    try:
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    return int(head.split(b" ", 2)[1])
