"""Credentials that ride in request URLs stay out of the logs and Sentry.

The browser's WebSocket carries the user's session JWT as ``?token=``, and some
paths are bearer secrets of their own. These start the REST and WebSocket
servers the way their entrypoints do, under uvicorn with the logging every
service process configures, and check what the requests leave behind.
"""

import asyncio
import io
import logging
import random
import re
import sys
import time
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
from starlette.requests import Request
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from backend.api import rest_api, ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.test_helpers import override_config
from backend.util import logging as service_logging
from backend.util import metrics
from backend.util.process import AppProcess
from backend.util.url_redaction import REDACTED, redact_url, redact_urls_in_text

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
    assert logs.count("[accepted]") == 2


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


@pytest.mark.parametrize(
    "url, expected",
    [
        ("/ws?token=a.b.c", f"/ws?token={REDACTED}"),
        ("/x?token=a&page=2&&flag", f"/x?token={REDACTED}&page={REDACTED}&&flag"),
        ("/x?token=a=b;c=d", f"/x?token={REDACTED}"),
        ("/x?a=1?token=b", f"/x?a={REDACTED}"),
        ("/x?=bare", f"/x?={REDACTED}"),
        # uvicorn's WebSocket scope keeps a raw # in the query.
        ("/ws?a=1#&token=b", f"/ws?a={REDACTED}&token={REDACTED}"),
        # Sentry builds the URL from the Host header, which can hold a ?.
        (
            "http://evil?/api/public/shared/chats/s-1/messages",
            f"http://evil?/api/public/shared/chats/{REDACTED}/messages",
        ),
        # A prefix stacked in front of a real one doesn't hide the real one.
        (
            "http://x/api/invitations/api/public/shared/chats/s-1/messages",
            f"http://x/api/invitations/{REDACTED}/public/shared/chats/{REDACTED}"
            "/messages",
        ),
        # The query is masked first, so its result is checked for paths too.
        (
            "/ws?/api/integrations/a=x/y&/webhooks/t",
            f"/ws?/api/integrations/a={REDACTED}&/webhooks/{REDACTED}",
        ),
        # Characters that can end a URL in a message don't end this one.
        ("/ws?q=it's&token=a", f"/ws?q={REDACTED}&token={REDACTED}"),
        ('/ws?q="&token=a', f"/ws?q={REDACTED}&token={REDACTED}"),
        ("/ws?q=a b&token=c", f"/ws?q={REDACTED}&token={REDACTED}"),
        (
            "https://h:1/api/email/unsubscribe?token=dXNlcjpzaWc%3D",
            f"https://h:1/api/email/unsubscribe?token={REDACTED}",
        ),
        (
            "/api/integrations/github/webhooks/w-1/ingress",
            f"/api/integrations/github/webhooks/{REDACTED}/ingress",
        ),
        (
            "/api/integrations/webhooks/w-1/ping",
            f"/api/integrations/webhooks/{REDACTED}/ping",
        ),
        (
            "/api/platform-linking/tokens/t-1/confirm",
            f"/api/platform-linking/tokens/{REDACTED}/confirm",
        ),
        (
            "/api/platform-linking/user-tokens/t-1/confirm",
            f"/api/platform-linking/user-tokens/{REDACTED}/confirm",
        ),
        ("/api/public/shared/s-1", f"/api/public/shared/{REDACTED}"),
        (
            "/api/public/shared/s-1/files/f-1/download",
            f"/api/public/shared/{REDACTED}/files/f-1/download",
        ),
        (
            "/api/public/shared/chats/s-1/messages?limit=5",
            f"/api/public/shared/chats/{REDACTED}/messages?limit={REDACTED}",
        ),
        ("/api/invitations/i-1/accept", f"/api/invitations/{REDACTED}/accept"),
        # A route that doesn't exist still carries the token.
        ("/api/invitations/i-1", f"/api/invitations/{REDACTED}"),
        ("/api/invitations/pending-1/x", f"/api/invitations/{REDACTED}/x"),
        # A path prefix in front of the app.
        (
            "/backend/api/invitations/i-1/accept",
            f"/backend/api/invitations/{REDACTED}/accept",
        ),
        # Already masked stays the same, and IDs that grant nothing are kept.
        (f"/ws?token={REDACTED}", f"/ws?token={REDACTED}"),
        ("/api/orgs/o-1/invitations/i-1", "/api/orgs/o-1/invitations/i-1"),
        ("/api/invitations/pending", "/api/invitations/pending"),
        ("/api/integrations/github/webhooks", "/api/integrations/github/webhooks"),
        ("/api/library/agents/a-1", "/api/library/agents/a-1"),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            '1.2.3.4:5 - "GET /api/invitations/i-1/decline?x=1 HTTP/1.1" 401',
            f'1.2.3.4:5 - "GET /api/invitations/{REDACTED}/decline?x={REDACTED}'
            ' HTTP/1.1" 401',
        ),
        # A raw quote is part of the query; only whitespace ends it.
        (
            '"GET /ws?q=it\'s&token=a HTTP/1.1"',
            f'"GET /ws?q={REDACTED}&token={REDACTED} HTTP/1.1"',
        ),
        (
            "acting as user u for requesting GET https://h/api/public/shared/s-1?x=1",
            "acting as user u for requesting GET "
            f"https://h/api/public/shared/{REDACTED}?x={REDACTED}",
        ),
        (
            '"GET /ws?a=1#&token=t HTTP/1.1"',
            f'"GET /ws?a={REDACTED}&token={REDACTED} HTTP/1.1"',
        ),
        ("Did it work? Yes.", "Did it work? Yes."),
    ],
)
def test_redact_urls_in_text(text: str, expected: str) -> None:
    assert redact_urls_in_text(text) == expected


def test_every_route_with_a_credential_in_its_path_is_masked_and_no_other() -> None:
    """Checks the masked prefixes against the REST app's real routes, so a new
    route with a token in its path fails here until it is covered."""
    for path in _route_paths(rest_api.app.routes):
        sample = _PATH_PARAM.sub(
            lambda match: (
                f"secret-{match[1]}" if match[1] in _CREDENTIAL_PARAMS else "value"
            ),
            path,
        )
        if _CREDENTIAL_PARAMS & set(_PATH_PARAM.findall(path)):
            assert "secret-" not in redact_url(sample), path
        else:
            assert redact_url(sample) == sample, path


@pytest.mark.parametrize("seed", range(5))
def test_no_request_target_keeps_its_credential_whatever_surrounds_it(
    seed: int,
) -> None:
    """Random request targets with the credential in a query value, in a path
    segment or both, among junk chosen to end a match early. Masked on their
    own and inside an access log line, the credential never survives, and
    masking again changes nothing."""
    rng = random.Random(seed)
    for _ in range(200):
        target = _random_request_target(rng)
        masked = redact_url(target)
        assert _SECRET not in masked, target
        assert redact_url(masked) == masked, target
        line = f'1.2.3.4:5 - "GET {target} HTTP/1.1" 200'
        masked_line = redact_urls_in_text(line)
        assert _SECRET not in masked_line, line
        assert redact_urls_in_text(masked_line) == masked_line, line


def test_masking_stays_linear_in_a_url_stacked_with_prefixes() -> None:
    """A client can send a URL made of thousands of prefixes, and the filter
    masks it on the event loop; quadratic work took seconds here."""
    url = "/api/invitations/x" * 50_000
    started = time.monotonic()
    assert redact_url(url) == f"/api/invitations/{REDACTED}" * 50_000
    assert time.monotonic() - started < 2


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


_PATH_PARAM = re.compile(r"{(\w+)(?::\w+)?}")
# The path parameters that are bearer credentials.
_CREDENTIAL_PARAMS = {"token", "share_token", "webhook_id"}


def _route_paths(routes: list[BaseRoute], prefix: str = "") -> Iterator[str]:
    for route in routes:
        if isinstance(route, Mount):
            yield from _route_paths(route.routes, prefix + route.path)
        elif isinstance(route, (Route, WebSocketRoute)):
            yield prefix + route.path


_SECRET = "SeCrEt-7f3a"
# A request target has no whitespace, and & always separates pairs; anything
# else may sit next to the credential, a raw # included.
_JUNK = ["", "a", "it's", '"', "?", "#", "=", ";", "/", "%26", "%3D", "[x]", "{"]
_CREDENTIAL_PATHS = [
    "/api/integrations/generic_webhook/webhooks/{}/ingress",
    "/api/integrations/webhooks/{}/ping",
    "/api/platform-linking/tokens/{}/confirm",
    "/api/platform-linking/user-tokens/{}/confirm",
    "/api/public/shared/{}",
    "/api/public/shared/chats/{}/messages",
    "/api/invitations/{}/accept",
    "/api/invitations/{}",
]


def _random_request_target(rng: random.Random) -> str:
    def junk() -> str:
        return "".join(rng.choices(_JUNK, k=rng.randint(0, 3)))

    pairs = [f"{junk()}={junk()}" for _ in range(rng.randint(0, 3))]
    if rng.random() < 0.5:
        pairs.insert(rng.randint(0, len(pairs)), f"token={junk()}{_SECRET}{junk()}")
        path = rng.choice(["/ws", "/api/email/unsubscribe", "/"])
    else:
        # A crafted path or Host header can stack another prefix in front.
        stacked = rng.choice(["", "", "/api/public/shared", "/api/invitations"])
        path = stacked + rng.choice(_CREDENTIAL_PATHS).format(_SECRET)
    # Sentry prefixes the scheme and the Host header, which the client controls.
    origin = f"http://h{junk()}" if rng.random() < 0.3 else ""
    query = "?" + "&".join(pairs) if pairs or rng.random() < 0.5 else ""
    return origin + path + query


class _CapturingTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self.payload = ""

    def capture_envelope(self, envelope: Envelope) -> None:
        self.payload += envelope.serialize().decode()
        self.events += [item.payload.json or {} for item in envelope.items]
