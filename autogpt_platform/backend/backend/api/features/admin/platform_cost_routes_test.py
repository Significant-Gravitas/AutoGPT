from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from .platform_cost_routes import router as platform_cost_router

app = fastapi.FastAPI()
app.include_router(platform_cost_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_get_dashboard_success(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_dashboard = AsyncMock(
        return_value=AsyncMock(
            by_provider=[],
            by_user=[],
            total_cost_microdollars=0,
            total_requests=0,
            total_users=0,
            model_dump=lambda **_: {
                "by_provider": [],
                "by_user": [],
                "total_cost_microdollars": 0,
                "total_requests": 0,
                "total_users": 0,
            },
        )
    )
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_dashboard",
        mock_dashboard,
    )

    response = client.get("/platform-costs/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "by_provider" in data
    assert "by_user" in data
    assert data["total_cost_microdollars"] == 0


def test_get_logs_success(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_logs",
        AsyncMock(return_value=([], 0)),
    )

    response = client.get("/platform-costs/logs")
    assert response.status_code == 200
    data = response.json()
    assert data["logs"] == []
    assert data["pagination"]["total_items"] == 0


def test_get_dashboard_with_filters(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_dashboard = AsyncMock(
        return_value=AsyncMock(
            by_provider=[],
            by_user=[],
            total_cost_microdollars=0,
            total_requests=0,
            total_users=0,
            model_dump=lambda **_: {
                "by_provider": [],
                "by_user": [],
                "total_cost_microdollars": 0,
                "total_requests": 0,
                "total_users": 0,
            },
        )
    )
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_dashboard",
        mock_dashboard,
    )

    response = client.get(
        "/platform-costs/dashboard",
        params={
            "start": "2026-01-01T00:00:00",
            "end": "2026-04-01T00:00:00",
            "provider": "openai",
            "user_id": "test-user-123",
        },
    )
    assert response.status_code == 200
    mock_dashboard.assert_called_once()
    call_kwargs = mock_dashboard.call_args.kwargs
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["user_id"] == "test-user-123"
    assert call_kwargs["start"] is not None
    assert call_kwargs["end"] is not None


def test_get_logs_with_pagination(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_logs",
        AsyncMock(return_value=([], 0)),
    )

    response = client.get(
        "/platform-costs/logs",
        params={"page": 2, "page_size": 25, "provider": "anthropic"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["pagination"]["current_page"] == 2
    assert data["pagination"]["page_size"] == 25


def test_get_dashboard_requires_admin() -> None:
    import fastapi
    from fastapi import HTTPException

    def reject_jwt(request: fastapi.Request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_jwt_payload] = reject_jwt
    try:
        response = client.get("/platform-costs/dashboard")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()
