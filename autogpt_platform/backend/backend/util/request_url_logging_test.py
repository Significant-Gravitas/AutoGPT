"""Credentials that ride in request URLs and headers stay out of the logs and
Sentry.

The browser's WebSocket carries the user's session JWT as ``?token=``, and some
paths are bearer secrets of their own. These start the REST and WebSocket
servers the way their entrypoints do, under uvicorn with the logging every
service process configures, and check what the requests leave behind.
"""

import asyncio
import io
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from autogpt_libs.auth import dependencies as auth_dependencies
from starlette.datastructures import URL
from starlette.requests import Request
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from backend.api import rest_api, ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.test_helpers import override_config
from backend.util.logging_test_helpers import CapturingTransport, serve
from backend.util.url_redaction import REDACTED

SESSION_TOKEN = "SECRET123"
SHARE_TOKEN = "5ec2e7ab-0000-4000-8000-00000000c0de"
WEBHOOK_ID = "0b5e55ed-0000-4000-8000-00000000beef"
INVITATION_TOKEN = "invitation-SECRET456"
CREDENTIALS = (SESSION_TOKEN, SHARE_TOKEN, WEBHOOK_ID, INVITATION_TOKEN)
SESSION_COOKIE = "cookie-SECRET789"
BEARER_TOKEN = "bearer-SECRET012"


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


@pytest.mark.parametrize("log_level", ["DEBUG"])
async def test_websocket_handshake_headers_stay_out_of_the_debug_log(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch, log_level: str
) -> None:
    """At DEBUG, websockets logs each handshake's target and headers: the
    server's on uvicorn's error log, the client's on its own."""
    async with _serve_websocket_server(monkeypatch) as host:
        with override_config(ws_api.settings, "enable_auth", False):
            async with connect(
                f"ws://{host}/ws?token={SESSION_TOKEN}",
                additional_headers={
                    "Cookie": f"session={SESSION_COOKIE}",
                    "Authorization": f"Bearer {BEARER_TOKEN}",
                },
            ):
                pass

    logs = service_output.getvalue()
    for credential in (SESSION_COOKIE, BEARER_TOKEN, SESSION_TOKEN):
        assert credential not in logs
    # The client logs what it sends, and the server what uvicorn hands it, with
    # the header names lowercased. Other headers are kept.
    for line in (
        f"> GET /ws?token={REDACTED} HTTP/1.1",
        f"> Cookie: {REDACTED}",
        f"> Authorization: {REDACTED}",
        f"< GET /ws?token={REDACTED} HTTP/1.1",
        f"< cookie: {REDACTED}",
        f"< authorization: {REDACTED}",
        f"< host: {host}",
    ):
        assert line in logs


async def test_rest_server_logs_no_credentials_from_request_urls(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
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
