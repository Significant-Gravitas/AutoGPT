from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.features.onboarding.routes import router
from backend.data.onboarding_wizard import OnboardingWizardConflict


async def authenticated_user() -> str:
    return "authenticated-user"


async def authenticated() -> None:
    return None


@pytest.mark.asyncio
async def test_draft_conflict_returns_409_for_authenticated_user(mocker):
    update = mocker.patch(
        "backend.api.features.onboarding.routes.update_user_onboarding",
        AsyncMock(side_effect=OnboardingWizardConflict()),
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[requires_user] = authenticated
    app.dependency_overrides[get_user_id] = authenticated_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/onboarding",
            json={
                "wizardProgress": None,
                "wizardRevision": 3,
                "wizardUserId": "authenticated-user",
                "userId": "another-user",
            },
        )

    assert response.status_code == 409
    assert "changed in another session" in response.json()["detail"]
    assert update.await_args.args[0] == "authenticated-user"


@pytest.mark.asyncio
async def test_draft_without_revision_returns_422_without_writing(mocker):
    update = mocker.patch(
        "backend.api.features.onboarding.routes.update_user_onboarding", AsyncMock()
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[requires_user] = authenticated
    app.dependency_overrides[get_user_id] = authenticated_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/onboarding",
            json={"wizardProgress": None, "wizardUserId": "authenticated-user"},
        )

    assert response.status_code == 422
    update.assert_not_awaited()
