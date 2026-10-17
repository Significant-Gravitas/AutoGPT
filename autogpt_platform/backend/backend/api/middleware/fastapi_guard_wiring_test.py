"""Tests for the optional fastapi-guard middleware wiring."""

import asyncio
import itertools
import os

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.middleware.fastapi_guard_wiring import (
    DEFAULT_EXCLUDED_PATHS,
    DEFAULT_TRUSTED_PROXIES,
    attach_guard,
    build_guard_config,
)

PRIVATE_RANGES = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]

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


def test_config_defaults():
    config = build_guard_config()
    assert config.enable_redis is False
    assert list(config.trusted_proxies) == PRIVATE_RANGES
    assert DEFAULT_TRUSTED_PROXIES.split(",") == PRIVATE_RANGES
    assert config.exclude_paths == DEFAULT_EXCLUDED_PATHS.split(",")
    assert config.enable_rate_limit_auto_ban is False
    assert config.security_headers is None
    assert config.enforce_https is False
    assert config.trust_x_forwarded_proto is False


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


@pytest.mark.parametrize("enabled", [False, True])
def test_app_service_builds_its_http_app(monkeypatch, enabled):
    from guard import SecurityMiddleware

    from backend.util.service import AppService

    class _Service(AppService):
        @classmethod
        def get_port(cls) -> int:
            return 0

    if enabled:
        monkeypatch.setenv("AUTOGPT_GUARD_ENABLED", "1")

    app = _Service().build_fastapi_app()

    has_guard = any(m.cls is SecurityMiddleware for m in app.user_middleware)
    assert has_guard is enabled
