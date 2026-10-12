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
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
import sentry_sdk
import uvicorn
from autogpt_libs.auth import dependencies as auth_dependencies
from sentry_sdk.envelope import Envelope
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.transport import Transport
from starlette.datastructures import URL
from starlette.requests import Request
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from backend.api import rest_api, ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.test_helpers import override_config
from backend.util import logging as service_logging
from backend.util import metrics
from backend.util.process import AppProcess
from backend.util.url_redaction import REDACTED

SESSION_TOKEN = "SECRET123"
SHARE_TOKEN = "5ec2e7ab-0000-4000-8000-00000000c0de"
WEBHOOK_ID = "0b5e55ed-0000-4000-8000-00000000beef"
INVITATION_TOKEN = "invitation-SECRET456"
CREDENTIALS = (SESSION_TOKEN, SHARE_TOKEN, WEBHOOK_ID, INVITATION_TOKEN)
_FAKE_DSN = "https://key@o1.ingest.us.sentry.io/1"


@pytest.fixture(params=[False, True], ids=["plain", "cloud"])
def service_output(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> io.StringIO:
    """Everything a service process writes to stdout and stderr at production's
    log level. ``cloud`` is the structured JSON Cloud Logging collects."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setattr(
        service_logging, "is_structured_logging_enabled", lambda: request.param
    )
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    service_logging.configure_logging()
    output = io.StringIO()
    for handler in root.handlers:
        assert isinstance(handler, logging.StreamHandler)
        handler.setStream(output)
    return output


@pytest.fixture
def sentry(monkeypatch: pytest.MonkeyPatch) -> Iterator["_CapturingTransport"]:
    """What a live SDK client, built from the options ``sentry_init()`` passes,
    would send to Sentry while the test runs."""
    calls: list[dict] = []
    with monkeypatch.context() as during_init:
        during_init.setattr(metrics, "_sentry_init", lambda **kw: calls.append(kw))
        during_init.setattr(metrics.settings.secrets, "sentry_dsn", _FAKE_DSN)
        during_init.delitem(sys.modules, "pytest")
        during_init.delenv("PYTEST_CURRENT_TEST", raising=False)
        metrics.sentry_init()
    [options] = calls
    # The profiler starts a sampler thread, and the asyncio integration patches
    # the session's shared event loop; neither sees request data.
    options["profiles_sample_rate"] = 0.0
    options["integrations"] = [
        integration
        for integration in options["integrations"]
        if not isinstance(integration, AsyncioIntegration)
    ]
    transport = _CapturingTransport()
    client = sentry_sdk.Client(**options, transport=transport)
    monkeypatch.setattr(sentry_sdk.get_global_scope(), "client", client)
    yield transport
    client.close()


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
    assert len(re.findall(r'\[REDACTED\]\\?" \[accepted\]', logs)) == 2


async def test_rest_server_logs_no_credentials_from_request_urls(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with (
        _serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        # Fails validation, so the 422 handler logs the request path.
        messages = await client.get(
            f"/api/public/shared/chats/{SHARE_TOKEN}/messages",
            params={"limit": "many", "token": SESSION_TOKEN},
        )
        # No webhook manager handles this provider, so the request fails
        # server-side and the 500 handler logs the request path.
        ingress = await client.post(
            f"/api/integrations/unregistered/webhooks/{WEBHOOK_ID}/ingress",
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
    webhook = f"/api/integrations/unregistered/webhooks/{REDACTED}/ingress"
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
    sentry: "_CapturingTransport", monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve_websocket_server(monkeypatch) as host:
        await _open_websockets(host)
    async with (
        _serve(rest_api.AgentServer(), monkeypatch) as host,
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


@asynccontextmanager
async def _serve(
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


@asynccontextmanager
async def _serve_websocket_server(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[str]:
    # Accepting a socket also subscribes it to the user's notifications in Redis.
    monkeypatch.setattr(
        ConnectionManager, "_start_notification_subscription", AsyncMock()
    )
    monkeypatch.setattr(ws_api, "_connection_manager", None)
    async with _serve(ws_api.WebsocketServer(), monkeypatch) as host:
        yield host


async def _open_websockets(host: str) -> None:
    url = f"ws://{host}/ws?token={SESSION_TOKEN}"
    # Not a valid JWT, so the handshake is refused, as an expired one's is...
    with pytest.raises(InvalidStatus):
        async with connect(url):
            pass
    # ...and with auth off it is accepted, as a live session's is. A raw quote
    # or # earlier in the query must not end the masking before the token.
    with override_config(ws_api.settings, "enable_auth", False):
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


class _CapturingTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self.payload = ""

    def capture_envelope(self, envelope: Envelope) -> None:
        self.payload += envelope.serialize().decode()
        self.events += [item.payload.json or {} for item in envelope.items]
