import base64
import hashlib
import hmac
import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

import httpx
import pytest
import uvicorn
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.util import process, service_auth
from backend.util.process import AppProcess
from backend.util.service import (
    AppService,
    AppServiceClient,
    HTTPClientError,
    endpoint_to_async,
    expose,
    get_service_client,
)
from backend.util.settings import Settings

KEY = "internal-key-for-tests-0123456789abcdef"
ENC_1 = base64.urlsafe_b64encode(b"1" * 32).decode()
ENC_2 = base64.urlsafe_b64encode(b"2" * 32).decode()
_port = 0


class KeyedService(AppService):
    @classmethod
    def get_port(cls) -> int:
        return _port

    @expose
    def add(self, a: int, b: int) -> int:
        return a + b


class KeyedServiceClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return KeyedService

    add = KeyedService.add
    add_async = endpoint_to_async(KeyedService.add)


class MetricsService(AppService):
    """Built only by the /metrics test: metrics register once per service name."""

    @classmethod
    def get_port(cls) -> int:
        return 0

    @expose
    def ping(self) -> str:
        return "pong"


class _Unstarted(AppProcess):
    reads_feature_flags = False
    ran = False

    def run(self) -> None:
        pass

    def execute_run_command(self, silent: bool) -> None:
        self.ran = True


class _SupportsClientCache(Protocol):
    _sync_clients: dict[str, httpx.Client]


def _settings(key: str = KEY, *, encryption_key: str = ""):
    settings = Settings()
    settings.secrets.internal_api_key = key
    settings.secrets.encryption_key = encryption_key
    return settings


@pytest.fixture
def use_settings(monkeypatch: pytest.MonkeyPatch):
    """Make services and their clients read these settings."""

    def use(settings: Settings) -> None:
        monkeypatch.setattr(service_auth, "Settings", lambda: settings)

    return use


def _app(use_settings, settings: Settings) -> FastAPI:
    use_settings(settings)
    return KeyedService().build_fastapi_app()


def _call_add(client: TestClient, **headers: str) -> httpx.Response:
    return client.post("/add", json={"a": 2, "b": 3}, headers=headers)


def _derived(encryption_key: str) -> str:
    """Pinned: changing the derivation makes every caller still running the
    previous release present a different key."""
    digest = hmac.new(
        base64.urlsafe_b64decode(encryption_key.encode()),
        b"autogpt internal service key v1",
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# Which key


def test_unset_the_key_is_derived_from_the_encryption_key():
    key = service_auth.internal_api_key(_settings("", encryption_key=ENC_1))

    assert key == _derived(ENC_1)
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", key)
    assert key != service_auth.internal_api_key(_settings("", encryption_key=ENC_2))


def test_a_set_key_wins_over_the_derived_one():
    settings = _settings(KEY, encryption_key=ENC_1)
    assert service_auth.internal_api_key(settings) == KEY


@pytest.mark.parametrize(
    "encryption_key",
    [
        "e",
        "changeme",
        "0" * 64,  # hex: 32 bytes written out, not encoded
        base64.urlsafe_b64encode(b"x" * 31).decode(),
        "not base64 at all!",
    ],
)
def test_no_key_is_derived_from_an_encryption_key_fernet_refuses(encryption_key):
    """A guessable ENCRYPTION_KEY would make the derived key guessable too."""
    with pytest.raises(ValueError):
        Fernet(encryption_key.encode())
    with pytest.raises(ValueError, match="ENCRYPTION_KEY is not a Fernet key"):
        service_auth.internal_api_key(_settings("", encryption_key=encryption_key))
    # A set key does not depend on it.
    assert service_auth.internal_api_key(_settings(KEY, encryption_key=encryption_key))


@pytest.mark.parametrize(
    "encryption_key",
    [
        Fernet.generate_key().decode(),
        base64.b64encode(b"\xfb\xff" * 16).decode(),  # standard alphabet: + and /
    ],
)
def test_any_key_fernet_accepts_derives_one(encryption_key):
    """No stricter than Fernet: an install whose credentials decrypt starts."""
    Fernet(encryption_key.encode())
    key = service_auth.internal_api_key(_settings("", encryption_key=encryption_key))
    assert key == _derived(encryption_key)


def test_every_encoding_of_one_key_derives_the_same_internal_key():
    """Services that decrypt the same credentials must agree on the key, even
    when one was given the key in another encoding Fernet also reads."""
    raw = b"\xfb\xff" * 16
    encodings = [
        base64.urlsafe_b64encode(raw).decode(),
        base64.b64encode(raw).decode(),  # standard alphabet: + and /
        base64.urlsafe_b64encode(raw).decode() + "\n",
    ]
    token = Fernet(encodings[0].encode()).encrypt(b"a credential")
    for encoding in encodings:
        assert Fernet(encoding.encode()).decrypt(token) == b"a credential"
    keys = {
        service_auth.internal_api_key(_settings("", encryption_key=encoding))
        for encoding in encodings
    }
    assert len(keys) == 1


@pytest.mark.parametrize(
    "malformed", ["short", "x" * 31, "a key with spaces that is long enough", "é" * 40]
)
def test_a_malformed_key_is_refused(malformed):
    with pytest.raises(ValueError, match="INTERNAL_API_KEY"):
        service_auth.internal_api_key(_settings(malformed))


@pytest.mark.parametrize(
    "settings, refused",
    [
        (_settings("short"), "INTERNAL_API_KEY"),
        (_settings("", encryption_key="changeme"), "ENCRYPTION_KEY"),
    ],
)
def test_a_bad_key_stops_every_process_starting(
    use_settings, monkeypatch, settings, refused
):
    monkeypatch.setattr(process, "check_secrets", lambda: None)
    use_settings(settings)
    unstarted = _Unstarted()

    with pytest.raises(ValueError, match=refused):
        unstarted.start()
    assert not unstarted.ran


def test_client_headers_carry_the_key():
    assert service_auth.internal_auth_headers(_settings()) == {
        "Authorization": f"Bearer {KEY}"
    }
    assert service_auth.internal_auth_headers(_settings("", encryption_key=ENC_1)) == {
        "Authorization": f"Bearer {_derived(ENC_1)}"
    }
    assert service_auth.internal_auth_headers(_settings("")) == {}


# What a service accepts


def test_an_exposed_method_needs_the_key(use_settings):
    client = TestClient(_app(use_settings, _settings()))

    assert _call_add(client).status_code == 401
    assert _call_add(client, Authorization="Bearer wrong-key").status_code == 401
    assert _call_add(client, Authorization=f"Basic {KEY}").status_code == 401
    assert _call_add(client, Authorization=KEY).status_code == 401
    assert _call_add(client, Authorization=f"Bearer  {KEY}").status_code == 401

    response = _call_add(client, Authorization=f"Bearer {KEY}")
    assert response.status_code == 200
    assert response.json() == 5


def test_the_derived_key_is_enforced(use_settings):
    client = TestClient(_app(use_settings, _settings("", encryption_key=ENC_1)))

    assert _call_add(client).status_code == 401
    assert _call_add(client, Authorization=f"Bearer {_derived(ENC_1)}").json() == 5


def test_the_key_is_checked_before_the_body_is_read(use_settings):
    """Unauthenticated bodies are never buffered or parsed: a malformed one is
    refused as unauthenticated, not reported as invalid."""
    client = TestClient(_app(use_settings, _settings()))
    json_type = {"content-type": "application/json"}
    with_key = {**json_type, "Authorization": f"Bearer {KEY}"}

    assert (
        client.post("/add", content=b"{not json", headers=json_type).status_code == 401
    )
    assert (
        client.post("/add", content=b"{not json", headers=with_key).status_code == 422
    )


async def _through_middleware(
    *,
    key: str = KEY,
    authorization: bytes | None = None,
    scope_type: str = "http",
) -> tuple[list[dict], bool]:
    """One request straight through the middleware: what it sent back, and
    whether the service behind it was reached. Reading the body fails."""
    reached = False

    async def app(scope, receive, send):
        nonlocal reached
        reached = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        raise AssertionError("the middleware read a request body")

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    middleware = service_auth.InternalKeyMiddleware(app, service_name="svc", key=key)
    headers = [] if authorization is None else [(b"authorization", authorization)]
    scope = {"type": scope_type, "path": "/add", "headers": headers}
    await middleware({**scope, "client": ("10.0.0.9", 1)}, receive, send)
    return sent, reached


async def test_the_middleware_never_reads_a_refused_body():
    sent, reached = await _through_middleware()

    assert not reached
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401


async def test_a_websocket_without_the_key_is_closed():
    sent, reached = await _through_middleware(scope_type="websocket")

    assert not reached
    assert sent == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.parametrize("authorization", [None, b"Bearer ", b"Bearer", b"Basic abc"])
async def test_without_a_key_not_even_an_empty_token_gets_through(authorization):
    sent, reached = await _through_middleware(key="", authorization=authorization)

    assert not reached
    assert sent[0]["status"] == 401


def test_a_non_ascii_key_is_refused_not_an_error(use_settings):
    client = TestClient(_app(use_settings, _settings()), raise_server_exceptions=False)
    response = client.post(
        "/add",
        json={"a": 2, "b": 3},
        headers={"Authorization": "Bearer clé".encode("latin-1")},
    )
    assert response.status_code == 401


def test_a_refusal_cannot_forge_a_log_line(use_settings, caplog):
    client = TestClient(_app(use_settings, _settings()))
    client.post("/add%0A2026-10-21 INFO forged", json={})

    assert "\n2026-10-21 INFO forged" not in caplog.text
    assert "\\n2026-10-21 INFO forged" in caplog.text


def test_health_checks_stay_open(use_settings):
    client = TestClient(_app(use_settings, _settings()))
    for path in ("/health_check", "/health_check_async"):
        assert client.get(path).status_code == 200
        assert client.post(path).status_code == 200


def test_metrics_stay_open(use_settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENABLE_METRICS", "true")
    use_settings(_settings())
    client = TestClient(MetricsService().build_fastapi_app())

    assert client.get("/metrics").status_code == 200
    assert client.post("/ping").status_code == 401


def test_the_api_docs_need_the_key(use_settings):
    client = TestClient(_app(use_settings, _settings()))
    for path in ("/openapi.json", "/docs"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": f"Bearer {KEY}"}).is_success


def test_without_any_key_a_service_refuses_every_call(use_settings, caplog):
    client = TestClient(_app(use_settings, _settings("")))

    assert _call_add(client).status_code == 401
    assert client.get("/health_check").status_code == 200
    assert "No internal service key" in caplog.text


# Real clients over a real socket


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def _serving(app: FastAPI) -> Iterator[int]:
    global _port
    _port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=_port, lifespan="off", log_config=None
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("test RPC server did not start")
        time.sleep(0.02)
    try:
        yield _port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextmanager
def _trap_proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[bytes]]:
    """An outbound proxy configured the usual way, recording what reaches it."""
    reached: list[bytes] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(2)
                try:
                    reached.append(conn.recv(1024))
                except OSError:
                    reached.append(b"")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    proxy = f"http://127.0.0.1:{listener.getsockname()[1]}"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    try:
        yield reached
    finally:
        stop.set()
        thread.join(timeout=2)
        listener.close()


@pytest.fixture
def local_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{KeyedService.__name__.upper()}_HOST", "127.0.0.1")


async def test_every_rpc_client_sends_the_key(use_settings, local_host):
    with _serving(_app(use_settings, _settings())) as port:
        async with httpx.AsyncClient() as raw:
            unauthenticated = await raw.post(
                f"http://127.0.0.1:{port}/add", json={"a": 2, "b": 3}
            )
        assert unauthenticated.status_code == 401

        client = get_service_client(KeyedServiceClient, request_retry=False)
        assert client.add(2, 3) == 5
        assert await client.add_async(2, 3) == 5

        # A client rebuilt after connection failures still carries it.
        cast(_SupportsClientCache, client)._sync_clients.clear()
        assert client.add(4, 5) == 9


def test_a_refused_caller_cannot_make_the_service_take_in_its_body(use_settings):
    """The connection closes with the refusal, so a body promised in the
    headers is never waited for or read."""
    with (
        _serving(_app(use_settings, _settings())) as port,
        socket.create_connection(("127.0.0.1", port), timeout=5) as conn,
    ):
        conn.sendall(
            b"POST /add HTTP/1.1\r\nHost: rpc\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100000000\r\n\r\n"
        )
        received = b""
        while chunk := conn.recv(4096):  # b"" once the service hangs up
            received += chunk

    assert received.startswith(b"HTTP/1.1 401")


def test_a_client_with_another_key_is_refused_and_not_retried(use_settings, local_host):
    with _serving(_app(use_settings, _settings())):
        use_settings(_settings("another-key-for-tests-0123456789abcdef"))
        client = get_service_client(KeyedServiceClient, request_retry=True)
        started = time.monotonic()
        with pytest.raises(HTTPClientError) as refused:
            client.add(2, 3)
        assert refused.value.status_code == 401
        assert time.monotonic() - started < 5


async def test_rpc_calls_never_go_through_an_environment_proxy(
    use_settings, monkeypatch
):
    """The calls are plain HTTP and carry the key: an outbound proxy set in the
    environment must not be where they go."""
    use_settings(_settings())
    # A host that never resolves: going direct fails fast, a proxy would be asked.
    monkeypatch.setenv(f"{KeyedService.__name__.upper()}_HOST", "keyed.invalid")
    with _trap_proxy(monkeypatch) as reached:
        client = get_service_client(KeyedServiceClient, request_retry=False)
        with pytest.raises(httpx.HTTPError):
            client.add(2, 3)
        with pytest.raises(httpx.HTTPError):
            await client.add_async(2, 3)
    assert reached == []
