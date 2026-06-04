from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.v1 import v1_router
from backend.data.onboarding import OnboardingStep

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize("invalid_step", ["VISIT_COPILOT", "BOGUS", "welcome", ""])
def test_complete_step_rejects_invalid_step(invalid_step):
    # Boundary validation is what replaces the dropped Prisma enum: any value
    # outside FrontendOnboardingStep must be rejected before any DB write. This
    # also locks in that the retired VISIT_COPILOT value is no longer accepted.
    response = client.post("/onboarding/step", params={"step": invalid_step})
    assert response.status_code == 422


def test_complete_step_accepts_renamed_complete_value(mocker):
    mock_complete = mocker.patch(
        "backend.api.features.v1.complete_onboarding_step",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.post(
        "/onboarding/step",
        params={"step": OnboardingStep.ONBOARDING_COMPLETE.value},
    )

    assert response.status_code == 200
    mock_complete.assert_awaited_once()
    assert mock_complete.await_args.args[1] == OnboardingStep.ONBOARDING_COMPLETE


def test_is_onboarding_completed_true_when_complete_step_present(mocker):
    from backend.data.model import UserOnboarding

    mock_get = mocker.patch(
        "backend.api.features.v1.get_user_onboarding",
        new_callable=AsyncMock,
    )
    mock_get.return_value = UserOnboarding.model_construct(
        completedSteps=[OnboardingStep.WELCOME, OnboardingStep.ONBOARDING_COMPLETE],
    )

    response = client.get("/onboarding/completed")

    assert response.status_code == 200
    assert response.json()["is_completed"] is True


def test_is_onboarding_completed_false_without_complete_step(mocker):
    from backend.data.model import UserOnboarding

    mock_get = mocker.patch(
        "backend.api.features.v1.get_user_onboarding",
        new_callable=AsyncMock,
    )
    mock_get.return_value = UserOnboarding.model_construct(
        completedSteps=[OnboardingStep.WELCOME],
    )

    response = client.get("/onboarding/completed")

    assert response.status_code == 200
    assert response.json()["is_completed"] is False
