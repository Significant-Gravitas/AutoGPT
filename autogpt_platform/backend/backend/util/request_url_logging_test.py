"""Credentials that ride in request URLs stay out of the logs and Sentry.

The browser's WebSocket carries the user's session JWT as ``?token=``, and some
paths are bearer secrets of their own. These start the REST and WebSocket
servers the way their entrypoints do, under uvicorn with the logging every
service process configures, and check what the requests leave behind.
"""

import asyncio
import io
import logging
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import uvicorn
from autogpt_libs.auth import dependencies as auth_dependencies
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.types import ASGIApp
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from backend.api import rest_api, ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.features.integrations import router as integrations_router
from backend.monitoring.instrumentation import METRICS_ENV_VAR
from backend.util.logging_test_helpers import CapturingTransport
from backend.util.process import AppProcess
from backend.util.url_redaction import REDACTED

SESSION_TOKEN = "SECRET123"
SHARE_TOKEN = "5ec2e7ab-0000-4000-8000-00000000c0de"
WEBHOOK_ID = "0b5e55ed-0000-4000-8000-00000000beef"
INVITATION_TOKEN = "invitation-SECRET456"
CREDENTIALS = (SESSION_TOKEN, SHARE_TOKEN, WEBHOOK_ID, INVITATION_TOKEN)


async def test_websocket_server_logs_no_session_token(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve_websocket_server(monkeypatch) as host:
        await _open_websockets(host)

    logs = service_output.getvalue()
    for credential in CREDENTIALS:
        assert credential not in logs
    assert logs.count(f"WebSocket /ws?token={REDACTED}") == 2
    assert logs.count(f"WebSocket /ws?q={REDACTED}&token={REDACTED}") == 1
    # The line keeps its shape, closing quote included (escaped in JSON).
    assert len(re.findall(r'\[REDACTED\]\\?" \[accepted\]', logs)) == 3


async def test_rest_server_logs_no_credentials_from_request_urls(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        integrations_router,
        "get_webhook",
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        # Fails validation, so the 422 handler logs the request path.
        messages = await client.get(
            f"/api/public/shared/chats/{SHARE_TOKEN}/messages",
            params={"limit": "many", "token": SESSION_TOKEN},
        )
        # The webhook lookup fails server-side, so the 500 handler logs the path.
        ingress = await client.post(
            f"/api/integrations/github/webhooks/{WEBHOOK_ID}/ingress",
            params={"token": SESSION_TOKEN},
            json={},
        )
    assert (messages.status_code, ingress.status_code) == (422, 500)

    logs = service_output.getvalue()
    for credential in CREDENTIALS:
        assert credential not in logs
    shared_chat = f"/api/public/shared/chats/{REDACTED}/messages"
    assert f"GET {shared_chat}?limit={REDACTED}&token={REDACTED} HTTP/1.1" in logs
    assert f"Validation failed for GET {shared_chat}" in logs
    webhook = f"/api/integrations/github/webhooks/{REDACTED}/ingress"
    assert f"POST {webhook}?token={REDACTED} HTTP/1.1" in logs
    assert f"POST {webhook} failed." in logs


async def test_admin_impersonation_audit_logs_no_credentials_from_the_url(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth_dependencies, "_heal_platform_user", AsyncMock())
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "server": ("backend", 443),
            "path": f"/api/invitations/{INVITATION_TOKEN}/accept",
            "query_string": f"token={SESSION_TOKEN}".encode(),
            "headers": [(b"host", b"backend"), (b"x-act-as-user-id", b"user-1")],
        }
    )
    admin = {"sub": "admin-1", "role": "admin", "email": "admin@example.com"}

    assert await auth_dependencies.get_user_id(request, admin) == "user-1"

    logs = service_output.getvalue()
    for credential in CREDENTIALS:
        assert credential not in logs
    url = f"https://backend/api/invitations/{REDACTED}/accept?token={REDACTED}"
    assert f"acting as user user-1 for requesting POST {url}" in logs


@pytest.mark.parametrize(
    "msg, args",
    [
        ('%s - "WebSocket %s" [accepted]', ("1.2.3.4:5", f"/ws?token={SESSION_TOKEN}")),
        ("GET /ws?token=%s", (SESSION_TOKEN,)),
        ("GET %(target)s", ({"target": f"/ws?token={SESSION_TOKEN}"},)),
        ("for requesting %s", (URL(f"https://backend/ws?token={SESSION_TOKEN}"),)),
    ],
    ids=["uvicorn", "split-across-format", "mapping", "url-object"],
)
def test_every_record_shape_on_a_filtered_logger_is_masked(
    service_output: io.StringIO, msg: str, args: tuple
) -> None:
    logging.getLogger("uvicorn.error").info(msg, *args)

    logs = service_output.getvalue()
    assert SESSION_TOKEN not in logs
    assert f"token={REDACTED}" in logs


async def test_sentry_gets_no_credentials_from_request_urls(
    sentry: CapturingTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve_websocket_server(monkeypatch) as host:
        await _open_websockets(host)
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        query = {"token": SESSION_TOKEN}
        await client.get(
            f"/api/public/shared/chats/{SHARE_TOKEN}/messages",
            params={"limit": "many", **query},
        )
        # Sent as %26, which Sentry decodes before before_send sees the query.
        await client.get(
            f"/api/public/shared/chats/{SHARE_TOKEN}/x",
            params={"token": f"&{SESSION_TOKEN}"},
        )
        # Sentry builds request.url from the Host header, which the client sets.
        await client.get(
            f"/api/public/shared/chats/{SHARE_TOKEN}/messages",
            params={"limit": "many"},
            headers={"host": "evil?"},
        )
        await client.get(
            f"/api/integrations/generic_webhook/webhooks/{WEBHOOK_ID}/ingress",
            params=query,
        )

    for credential in CREDENTIALS:
        assert credential not in sentry.payload
    requests = [event["request"] for event in sentry.events if "request" in event]
    assert len(requests) >= 6
    assert {request.get("query_string") for request in requests} == {REDACTED}
    assert any(
        f"/api/public/shared/chats/{REDACTED}/messages" in request["url"]
        for request in requests
    )
    # The request no route matched is named after its URL.
    assert f"/api/public/shared/chats/{REDACTED}/x" in {
        event.get("transaction", "").removeprefix(f"http://{host}")
        for event in sentry.events
    }


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
        for _ in range(1000):
            if server.started or serving.done():
                break
            await asyncio.sleep(0.01)
        assert server.started, "uvicorn did not start"
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"127.0.0.1:{port}"
    finally:
        if server.started:
            server.should_exit = True
        else:
            # uvicorn reads should_exit only once it has started.
            serving.cancel()
        try:
            await asyncio.wait({serving})
        except asyncio.CancelledError:
            # Cancelled while it shuts down: a server that has started finishes
            # its bounded shutdown, closing its listener, and one still starting
            # is cancelled, so neither is left running.
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


@asynccontextmanager
async def _serve_websocket_server(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[str]:
    # Accepting a socket also subscribes it to the user's notifications in Redis.
    monkeypatch.setattr(
        ConnectionManager, "_start_notification_subscription", AsyncMock()
    )
    monkeypatch.setattr(ws_api, "_connection_manager", None)
    async with serve(ws_api.WebsocketServer(), monkeypatch) as host:
        yield host


async def _open_websockets(host: str) -> None:
    url = f"ws://{host}/ws?token={SESSION_TOKEN}"
    async with connect(url) as websocket:
        with pytest.raises(ConnectionClosedError) as closed:
            await asyncio.wait_for(websocket.recv(), timeout=3)
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == 4003
    # Once authenticated, a raw quote or # must not end query-token masking.
    with patch.object(
        ws_api, "authenticate_websocket", AsyncMock(return_value="user-1")
    ):
        async with connect(url):
            pass
        target = f"/ws?q=it's#&token={SESSION_TOKEN}"
        assert b" 101 " in await _raw_websocket_handshake(host, target)


async def _raw_websocket_handshake(host: str, target: str) -> bytes:
    """Sent as bytes, since a client library drops a # before sending."""
    address, port = host.split(":")
    reader, writer = await asyncio.open_connection(address, int(port))
    writer.write(
        f"GET {target} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n".encode()
    )
    status_line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return status_line
