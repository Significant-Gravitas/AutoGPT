"""Tests for the optional fastapi-guard middleware wiring."""

import asyncio
import itertools
import os

import pytest
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.routing import APIWebSocketRoute
from fastapi.testclient import TestClient
from guard import SecurityMiddleware
from guard.websocket import WS_CLOSE_IP_NOT_ALLOWED
from httpx import ASGITransport, AsyncClient

from backend.api.middleware.fastapi_guard_wiring import (
    DEFAULT_EXCLUDED_PATHS,
    attach_guard,
    build_guard_config,
    guard_websocket_connection,
)
from backend.api.ws_api import app as websocket_app
from backend.util.service import AppService

# Rate limiting and ban state in fastapi-guard are process-wide, so every
# test drives its own TEST-NET client IP to stay hermetic.
_IP = itertools.count(1)


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


@pytest.fixture(autouse=True)
def _clean_guard_env(monkeypatch):
    for name in [k for k in os.environ if k.startswith("AUTOGPT_GUARD_")]:
        monkeypatch.delenv(name)


def _run(monkeypatch, scenario, client_ip, **env):
    """Build a fresh app with env overrides and run an async scenario."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    attach_guard(app)

    async def runner():
        transport = ASGITransport(app=app, client=(client_ip, 50000))
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            await scenario(client)

    asyncio.run(runner())


def test_disabled_by_default():
    app = FastAPI()
    attach_guard(app)
    assert len(app.user_middleware) == 0
    assert not hasattr(app.state, "guard_decorator")


def test_enabled_sets_guard_decorator(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_ENABLED", "1")
    app = FastAPI()
    attach_guard(app)
    assert len(app.user_middleware) == 1
    assert app.state.guard_decorator is not None


def test_blocked_ip_is_rejected(monkeypatch):
    blocked = _unique_ip()

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 403

    _run(
        monkeypatch,
        scenario,
        blocked,
        AUTOGPT_GUARD_ENABLED="1",
        AUTOGPT_GUARD_BLOCKED_IPS=blocked,
    )


def test_rate_limit_returns_429(monkeypatch):
    client_ip = _unique_ip()

    async def scenario(client):
        assert (await client.get("/ping")).status_code == 200
        assert (await client.get("/ping")).status_code == 200
        assert (await client.get("/ping")).status_code == 429

    _run(
        monkeypatch,
        scenario,
        client_ip,
        AUTOGPT_GUARD_ENABLED="1",
        AUTOGPT_GUARD_RATE_LIMIT="2",
        AUTOGPT_GUARD_RATE_LIMIT_WINDOW="60",
    )


def test_mounted_app_counts_once_against_the_outer_guard(monkeypatch):
    # The external API is mounted inside the REST app and is not guarded on
    # its own: the REST app's middleware already sees every mounted request.
    monkeypatch.setenv("AUTOGPT_GUARD_ENABLED", "1")
    monkeypatch.setenv("AUTOGPT_GUARD_RATE_LIMIT", "2")
    mounted = FastAPI()

    @mounted.get("/ping")
    async def ping():
        return {"ok": True}

    app = FastAPI()
    attach_guard(app)
    app.mount("/external-api", mounted)

    async def runner():
        transport = ASGITransport(app=app, client=(_unique_ip(), 50000))
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            assert (await client.get("/external-api/ping")).status_code == 200
            assert (await client.get("/external-api/ping")).status_code == 200
            assert (await client.get("/external-api/ping")).status_code == 429

    asyncio.run(runner())


def test_config_defaults():
    config = build_guard_config()
    assert config.enable_redis is False
    assert tuple(config.trusted_proxies) == ()
    assert config.enable_ip_banning is True
    assert config.enable_penetration_detection is True
    assert config.detection_scan_body is False
    assert config.exclude_paths == DEFAULT_EXCLUDED_PATHS.split(",")
    assert config.enable_rate_limit_auto_ban is False
    assert config.security_headers is None
    assert config.enforce_https is False
    assert config.trust_x_forwarded_proto is False


def test_detection_toggles(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_IP_BANNING", "0")
    monkeypatch.setenv("AUTOGPT_GUARD_PENETRATION_DETECTION", "false")
    monkeypatch.setenv("AUTOGPT_GUARD_SCAN_BODY", "1")
    config = build_guard_config()
    assert config.enable_ip_banning is False
    assert config.enable_penetration_detection is False
    assert config.detection_scan_body is True


def test_explicit_empty_trusted_proxies_trusts_none(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_TRUSTED_PROXIES", "")
    assert tuple(build_guard_config().trusted_proxies) == ()


def test_explicit_trusted_proxies(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_TRUSTED_PROXIES", "203.0.113.5, 203.0.113.0/24")
    assert list(build_guard_config().trusted_proxies) == [
        "203.0.113.5",
        "203.0.113.0/24",
    ]


def test_enforce_https_derives_trust_x_forwarded_proto(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_ENFORCE_HTTPS", "1")
    config = build_guard_config()
    assert config.enforce_https is True
    assert config.trust_x_forwarded_proto is True


def test_explicit_trust_x_forwarded_proto_overrides_derivation(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_ENFORCE_HTTPS", "1")
    monkeypatch.setenv("AUTOGPT_GUARD_TRUST_X_FORWARDED_PROTO", "0")
    assert build_guard_config().trust_x_forwarded_proto is False


def test_security_headers_enabled(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_SECURITY_HEADERS", "true")
    headers = build_guard_config().security_headers
    assert headers is not None
    assert headers["enabled"] is True


def test_lists_and_geo_options(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_ALLOWED_IPS", "203.0.113.7")
    monkeypatch.setenv("AUTOGPT_GUARD_BLOCKED_COUNTRIES", "XX,YY")
    monkeypatch.setenv("AUTOGPT_GUARD_ALLOWED_COUNTRIES", "US")
    monkeypatch.setenv("AUTOGPT_GUARD_BLOCK_CLOUD_PROVIDERS", "AWS,GCP")
    monkeypatch.setenv("AUTOGPT_GUARD_IPINFO_TOKEN", "test-token")
    # guard warns that blocked_countries is moot next to whitelist_countries
    with pytest.warns(UserWarning, match="blocked_countries"):
        config = build_guard_config()
    assert list(config.whitelist) == ["203.0.113.7"]
    assert set(config.blocked_countries) == {"XX", "YY"}
    assert set(config.whitelist_countries) == {"US"}
    assert set(config.block_cloud_providers) == {"AWS", "GCP"}
    assert config.ipinfo_token == "test-token"


def test_redis_url_enables_redis(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_REDIS_URL", "redis://localhost:6379")
    config = build_guard_config()
    assert config.enable_redis is True
    assert config.redis_url == "redis://localhost:6379"
    assert config.redis_prefix == "autogpt_guard:"


def test_invalid_boolean_raises(monkeypatch):
    monkeypatch.setenv("AUTOGPT_GUARD_ENABLED", "tru")
    with pytest.raises(ValueError, match="AUTOGPT_GUARD_ENABLED"):
        attach_guard(FastAPI())


def _websocket_app() -> FastAPI:
    app = FastAPI()
    attach_guard(app)

    @app.websocket("/ws", dependencies=[Depends(guard_websocket_connection)])
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return app


def test_websocket_handshake_is_guarded(monkeypatch):
    blocked = _unique_ip()
    monkeypatch.setenv("AUTOGPT_GUARD_ENABLED", "1")
    monkeypatch.setenv("AUTOGPT_GUARD_BLOCKED_IPS", blocked)
    app = _websocket_app()

    with (
        TestClient(app, client=(blocked, 50000)) as client,
        pytest.raises(WebSocketDisconnect) as closed,
        client.websocket_connect("/ws"),
    ):
        pass
    assert closed.value.code == WS_CLOSE_IP_NOT_ALLOWED.code

    with (
        TestClient(app, client=(_unique_ip(), 50000)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        assert websocket.receive_text() == "hello"


def test_websocket_server_route_runs_the_guard_check():
    route = next(
        r
        for r in websocket_app.routes
        if isinstance(r, APIWebSocketRoute) and r.path == "/ws"
    )
    calls = [dep.call for dep in route.dependant.dependencies]
    assert guard_websocket_connection in calls


def test_websocket_dependency_is_a_no_op_when_disabled():
    with (
        TestClient(_websocket_app(), client=(_unique_ip(), 50000)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        assert websocket.receive_text() == "hello"


@pytest.mark.parametrize(
    "env, guarded",
    [
        ({}, False),
        ({"AUTOGPT_GUARD_ENABLED": "1"}, False),
        (
            {"AUTOGPT_GUARD_ENABLED": "1", "AUTOGPT_GUARD_INTERNAL_SERVICES": "1"},
            True,
        ),
    ],
)
def test_app_service_is_guarded_only_on_its_own_opt_in(monkeypatch, env, guarded):
    class _Service(AppService):
        @classmethod
        def get_port(cls) -> int:
            return 0

    for name, value in env.items():
        monkeypatch.setenv(name, value)

    app = _Service().build_fastapi_app()

    has_guard = any(m.cls is SecurityMiddleware for m in app.user_middleware)
    assert has_guard is guarded
