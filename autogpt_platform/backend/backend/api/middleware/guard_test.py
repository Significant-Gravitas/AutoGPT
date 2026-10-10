"""Tests for the optional fastapi-guard middleware wiring."""

import asyncio
import itertools

import pytest
from fastapi import FastAPI

from backend.api.middleware.guard import attach_guard

pytest.importorskip("guard")

from httpx import ASGITransport, AsyncClient  # noqa: E402

# Rate limiting and ban state in fastapi-guard are process-wide, so every
# test drives its own TEST-NET client IP to stay hermetic.
_IP = itertools.count(1)


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


def _run(scenario, client_ip, **env):
    """Build a fresh app with env overrides and run an async scenario."""
    import os

    old = {}
    for name, value in env.items():
        old[name] = os.environ.get(name)
        os.environ[name] = value

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

    try:
        asyncio.run(runner())
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AUTOGPT_GUARD_ENABLED", raising=False)
    app = FastAPI()
    attach_guard(app)
    assert len(app.user_middleware) == 0


def test_blocked_ip_is_rejected(monkeypatch):
    blocked = _unique_ip()

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 403

    _run(
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
        scenario,
        client_ip,
        AUTOGPT_GUARD_ENABLED="1",
        AUTOGPT_GUARD_RATE_LIMIT="2",
        AUTOGPT_GUARD_RATE_LIMIT_WINDOW="60",
    )
