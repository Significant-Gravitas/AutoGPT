import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.util import service_auth
from backend.util.exceptions import MissingConfigError
from backend.util.service import (
    AppService,
    AppServiceClient,
    HTTPClientError,
    endpoint_to_async,
    expose,
    get_service_client,
)
from backend.util.settings import BehaveAs, Settings

KEY = "internal-key-for-tests"
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


class _SupportsClientCache(Protocol):
    _sync_clients: dict[str, httpx.Client]


def _settings(key: str = KEY, *, cloud: bool = False, required: bool = False):
    settings = Settings()
    settings.secrets.internal_api_key = key
    settings.config.behave_as = BehaveAs.CLOUD if cloud else BehaveAs.LOCAL
    settings.config.internal_rpc_require = required
    return settings


@pytest.fixture
def use_settings(monkeypatch: pytest.MonkeyPatch):
    """Make the server and its clients read these settings."""

    def use(settings: Settings) -> None:
        monkeypatch.setattr(service_auth, "Settings", lambda: settings)

    return use


def _app(use_settings, settings: Settings) -> FastAPI:
    use_settings(settings)
    return KeyedService().build_fastapi_app()


def _call_add(client: TestClient, **headers: str) -> httpx.Response:
    return client.post("/add", json={"a": 2, "b": 3}, headers=headers)


# Where the key is enforced


def test_local_with_a_key_enforces_it():
    assert len(service_auth.internal_auth_dependencies("svc", _settings())) == 1


def test_cloud_sends_the_key_without_requiring_it_until_told(caplog):
    assert service_auth.internal_auth_dependencies("svc", _settings(cloud=True)) == []
    assert "not yet required" in caplog.text


def test_cloud_enforces_once_required():
    settings = _settings(cloud=True, required=True)
    assert len(service_auth.internal_auth_dependencies("svc", settings)) == 1


def test_no_key_accepts_calls_and_warns(caplog):
    assert service_auth.internal_auth_dependencies("svc", _settings("")) == []
    assert "INTERNAL_API_KEY is not set" in caplog.text


def test_required_without_a_key_refuses_to_start(use_settings):
    use_settings(_settings("", required=True))
    with pytest.raises(MissingConfigError, match="INTERNAL_API_KEY"):
        KeyedService().build_fastapi_app()


def test_client_headers_carry_the_key_only_when_set():
    assert service_auth.internal_auth_headers(_settings()) == {
        "Authorization": f"Bearer {KEY}"
    }
    assert service_auth.internal_auth_headers(_settings("")) == {}


# What an enforcing service accepts


def test_an_exposed_method_needs_the_key(use_settings):
    client = TestClient(_app(use_settings, _settings()))

    assert _call_add(client).status_code == 401
    assert _call_add(client, Authorization="Bearer wrong-key").status_code == 401
    assert _call_add(client, Authorization=f"Basic {KEY}").status_code == 401
    assert _call_add(client, Authorization=KEY).status_code == 401

    response = _call_add(client, Authorization=f"Bearer {KEY}")
    assert response.status_code == 200
    assert response.json() == 5


def test_a_non_ascii_key_is_refused_not_an_error(use_settings):
    client = TestClient(_app(use_settings, _settings()), raise_server_exceptions=False)
    response = client.post(
        "/add",
        json={"a": 2, "b": 3},
        headers={"Authorization": "Bearer clé".encode("latin-1")},
    )
    assert response.status_code == 401


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


def test_cloud_accepts_calls_without_the_key_until_required(use_settings):
    client = TestClient(_app(use_settings, _settings(cloud=True)))
    assert _call_add(client).status_code == 200


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


def test_a_client_without_the_key_is_refused_and_not_retried(use_settings, local_host):
    app = _app(use_settings, _settings())
    with _serving(app):
        use_settings(_settings(""))
        client = get_service_client(KeyedServiceClient, request_retry=True)
        started = time.monotonic()
        with pytest.raises(HTTPClientError) as refused:
            client.add(2, 3)
        assert refused.value.status_code == 401
        assert time.monotonic() - started < 5
