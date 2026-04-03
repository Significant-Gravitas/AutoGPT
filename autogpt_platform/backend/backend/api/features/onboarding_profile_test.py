from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.v1 import v1_router

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_onboarding_profile_success(mocker):
    mock_extract = mocker.patch(
        "backend.api.features.v1.extract_business_understanding",
        new_callable=AsyncMock,
    )
    mock_upsert = mocker.patch(
        "backend.api.features.v1.upsert_business_understanding",
        new_callable=AsyncMock,
    )

    from backend.data.understanding import BusinessUnderstandingInput

    mock_extract.return_value = BusinessUnderstandingInput.model_construct(
        user_name="John",
        user_role="Founder/CEO",
        pain_points=["Finding leads"],
        suggested_prompts={"Learn": ["How do I automate lead gen?"]},
    )
    mock_upsert.return_value = AsyncMock()

    response = client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": "Founder/CEO",
            "pain_points": ["Finding leads", "Email & outreach"],
        },
    )
    assert response.status_code == 200
    mock_extract.assert_awaited_once()
    mock_upsert.assert_awaited_once()


def test_onboarding_profile_missing_fields():
    response = client.post(
        "/onboarding/profile",
        json={"user_name": "John"},
    )
    assert response.status_code == 422
