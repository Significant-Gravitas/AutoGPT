import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from autogpt_libs.autogpt_libs.rate_limit.middleware import rate_limit_middleware


@pytest.fixture
def app():
    app = FastAPI()
    app.middleware("http")(rate_limit_middleware)

    @app.get("/")
    def root():
        return {"ok": True}

    @app.get("/api/items")
    def api_items():
        return {"items": []}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class FakeAllowedLimiter:
    def __init__(self, *args, **kwargs):
        self.max_requests = 100

    async def check_rate_limit(self, api_key: str):
        # allowed, remaining, reset_time
        return True, 99, 1700000000


class FakeBlockedLimiter:
    def __init__(self, *args, **kwargs):
        self.max_requests = 100

    async def check_rate_limit(self, api_key: str):
        # not allowed, remaining, reset_time
        return False, 0, 1700000000


def test_non_api_paths_skipped(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers
    assert "X-RateLimit-Remaining" not in resp.headers
    assert "X-RateLimit-Reset" not in resp.headers


def test_missing_auth_skips_rate_limit(client):
    resp = client.get("/api/items")
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers
    assert "X-RateLimit-Remaining" not in resp.headers
    assert "X-RateLimit-Reset" not in resp.headers


def test_allowed_sets_headers(client, monkeypatch):
    monkeypatch.setattr(
        "autogpt_libs.autogpt_libs.rate_limit.middleware.RateLimiter",
        FakeAllowedLimiter,
    )
    resp = client.get("/api/items", headers={"Authorization": "Bearer key123"})
    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "100"
    assert resp.headers["X-RateLimit-Remaining"] == "99"
    assert resp.headers["X-RateLimit-Reset"] == "1700000000"


def test_blocked_raises_429(client, monkeypatch):
    monkeypatch.setattr(
        "autogpt_libs.autogpt_libs.rate_limit.middleware.RateLimiter",
        FakeBlockedLimiter,
    )
    resp = client.get("/api/items", headers={"Authorization": "Bearer key123"})
    assert resp.status_code == 429
    assert resp.json()["detail"] == "Rate limit exceeded. Please try again later."
