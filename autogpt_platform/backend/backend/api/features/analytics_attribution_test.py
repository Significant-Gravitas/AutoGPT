"""Tests for the user attribution endpoint."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.data.attribution import UserAttribution, UserAttributionInput

from .analytics import router as analytics_router

app = fastapi.FastAPI()
app.include_router(analytics_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _stored(user_id: str, data: UserAttributionInput) -> UserAttribution:
    return UserAttribution(
        user_id=user_id,
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
        **data.model_dump(),
    )


def test_attribution_merges_datafast_headers_into_body(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    record = mocker.patch(
        "backend.api.features.analytics.attribution_db.record_user_attribution",
        new_callable=AsyncMock,
        side_effect=lambda user_id, data: _stored(user_id, data),
    )

    response = client.post(
        "/attribution",
        json={
            "anonymous_id": "anon-1",
            "landing_path": "/marketplace?utm_source=x",
            "utm_source": "x",
            "signup_method": "google",
        },
        headers={
            "X-Datafast-Visitor-Id": "vis-1",
            "X-Datafast-Session-Id": "ses-1",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_id"] == test_user_id
    assert body["datafast_visitor_id"] == "vis-1"
    assert body["datafast_session_id"] == "ses-1"
    assert body["anonymous_id"] == "anon-1"
    assert body["signup_method"] == "google"
    record.assert_awaited_once()
    sent: UserAttributionInput = record.call_args.args[1]
    assert sent.datafast_visitor_id == "vis-1"
    assert sent.utm_source == "x"


def test_attribution_body_wins_over_headers(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    record = mocker.patch(
        "backend.api.features.analytics.attribution_db.record_user_attribution",
        new_callable=AsyncMock,
        side_effect=lambda user_id, data: _stored(user_id, data),
    )

    response = client.post(
        "/attribution",
        json={"datafast_visitor_id": "vis-body"},
        headers={"X-Datafast-Visitor-Id": "vis-header"},
    )

    assert response.status_code == 200, response.text
    assert record.call_args.args[1].datafast_visitor_id == "vis-body"


def test_attribution_rejects_oversized_fields() -> None:
    response = client.post("/attribution", json={"utm_source": "x" * 300})
    assert response.status_code == 422
