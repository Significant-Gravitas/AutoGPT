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

    response = client.get("/admin/platform_costs/dashboard")
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

    response = client.get("/admin/platform_costs/logs")
    assert response.status_code == 200
    data = response.json()
    assert data["logs"] == []
    assert data["pagination"]["total_items"] == 0


def test_get_dashboard_requires_admin() -> None:
    app.dependency_overrides.clear()
    response = client.get("/admin/platform_costs/dashboard")
    assert response.status_code in (401, 403)
