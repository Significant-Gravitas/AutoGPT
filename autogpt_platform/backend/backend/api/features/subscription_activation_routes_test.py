"""Authenticated activation API, explicit consent, and recoverable failures."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.features import subscription_activation_routes as routes
from backend.data.subscription_activation_models import (
    ActivationNotFound,
    ActivationResponse,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[routes.enforce_subscription_status_rate_limit] = (
        allow_request
    )
    return app


def authorize(app):
    app.dependency_overrides[routes.requires_user] = allow_request
    app.dependency_overrides[routes.get_user_id] = authenticated_user


async def allow_request():
    return None


async def authenticated_user():
    return "authenticated-user"


@pytest.mark.asyncio
async def test_activation_endpoints_require_authentication(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://app.test"
    ) as client:
        for method, path in [
            ("GET", "/current"),
            ("GET", "/attempt"),
            ("POST", "/preview"),
            ("POST", "/attempt/confirm"),
        ]:
            response = await client.request(
                method, f"/api/credits/pro-activation{path}", json={}
            )
            assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_confirm_rejects_missing_or_false_consent(app, monkeypatch):
    authorize(app)
    confirm = AsyncMock()
    monkeypatch.setattr(routes, "confirm_activation", confirm)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://app.test"
    ) as client:
        for body in [{}, {"confirmed": False, "terms_token": "a" * 64}]:
            response = await client.post(
                "/api/credits/pro-activation/attempt/confirm", json=body
            )
            assert response.status_code == 422
    confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_owned_attempt_read_passes_authenticated_user_and_returns_404(
    app, monkeypatch
):
    authorize(app)
    get = AsyncMock(side_effect=ActivationNotFound("Activation was not found"))
    monkeypatch.setattr(routes, "get_activation", get)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://app.test"
    ) as client:
        response = await client.get("/api/credits/pro-activation/someone-elses-attempt")
        assert response.status_code == 404
    get.assert_awaited_once_with("authenticated-user", "someone-elses-attempt")


@pytest.mark.asyncio
async def test_current_failure_returns_processing_without_retrying_payment(
    app, monkeypatch
):
    authorize(app)
    monkeypatch.setattr(
        routes, "current_activation", AsyncMock(side_effect=ConnectionError("DB down"))
    )
    confirm = AsyncMock()
    monkeypatch.setattr(routes, "confirm_activation", confirm)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://app.test"
    ) as client:
        response = await client.get("/api/credits/pro-activation/current")
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["retry_after_seconds"] == 3
    confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_current_ready_preserves_return_destination(app, monkeypatch):
    authorize(app)
    monkeypatch.setattr(
        routes,
        "current_activation",
        AsyncMock(
            return_value=ActivationResponse(
                status="ready", return_to="/chat/123?resume=1"
            )
        ),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://app.test"
    ) as client:
        response = await client.get("/api/credits/pro-activation/current")
    assert response.json()["return_to"] == "/chat/123?resume=1"
