from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.onboarding.routes import router
from backend.data.onboarding_role import OnboardingRole

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

ROUTES = "backend.api.features.onboarding.routes"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def profile(mocker):
    from backend.data.understanding import BusinessUnderstandingInput

    mocks = {
        name: mocker.patch(f"{ROUTES}.{name}", new_callable=AsyncMock)
        for name in (
            "extract_business_understanding",
            "upsert_business_understanding",
            "save_onboarding_role",
            "queue_onboarding_role",
        )
    }
    mocks["set_onboarding_role"] = mocker.patch(
        f"{ROUTES}.set_onboarding_role", new_callable=MagicMock
    )
    # The extraction's own idea of the role, which AutoPilot-side rewrites
    # look like; the kept pick comes from the request.
    mocks["extract_business_understanding"].return_value = (
        BusinessUnderstandingInput.model_construct(
            user_name="John",
            user_role="decision maker",
            pain_points=["Finding leads"],
            suggested_prompts={"Learn": ["How do I automate lead gen?"]},
        )
    )
    return mocks


def _submit(user_role: str):
    return client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": user_role,
            "pain_points": ["Finding leads", "Email & outreach"],
        },
    )


def test_onboarding_profile_success(profile):
    response = _submit("Founder/CEO")
    assert response.status_code == 200
    profile["extract_business_understanding"].assert_awaited_once()
    profile["upsert_business_understanding"].assert_awaited_once()


@pytest.mark.parametrize(
    "user_role, role",
    [
        ("Founder/CEO", OnboardingRole(choice="Founder/CEO")),
        ("Dentist", OnboardingRole(choice="Other", other="Dentist")),
    ],
)
def test_the_pick_is_kept_and_sent_to_mailerlite_and_posthog(
    profile, test_user_id, user_role, role
):
    assert _submit(user_role).status_code == 200
    profile["save_onboarding_role"].assert_awaited_once_with(test_user_id, role)
    profile["queue_onboarding_role"].assert_awaited_once_with(test_user_id, role)
    profile["set_onboarding_role"].assert_called_once_with(
        user_id=test_user_id, role=role
    )


def test_onboarding_profile_missing_fields():
    response = client.post(
        "/onboarding/profile",
        json={"user_name": "John"},
    )
    assert response.status_code == 422
