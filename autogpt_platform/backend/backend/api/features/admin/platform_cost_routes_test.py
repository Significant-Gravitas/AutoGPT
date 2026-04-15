from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.data.platform_cost import CostLogRow, PlatformCostDashboard

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
    real_dashboard = PlatformCostDashboard(
        by_provider=[],
        by_user=[],
        total_cost_microdollars=0,
        total_requests=0,
        total_users=0,
    )
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_dashboard",
        AsyncMock(return_value=real_dashboard),
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
    real_dashboard = PlatformCostDashboard(
        by_provider=[],
        by_user=[],
        total_cost_microdollars=0,
        total_requests=0,
        total_users=0,
    )
    mock_dashboard = AsyncMock(return_value=real_dashboard)
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
        response = client.get("/platform-costs/logs")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_get_dashboard_rejects_non_admin(mock_jwt_user, mock_jwt_admin) -> None:
    """Non-admin JWT must be rejected with 403 by requires_admin_user."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    try:
        response = client.get("/platform-costs/dashboard")
        assert response.status_code == 403
        response = client.get("/platform-costs/logs")
        assert response.status_code == 403
    finally:
        app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]


def test_get_logs_invalid_page_size_too_large() -> None:
    """page_size > 200 must be rejected with 422."""
    response = client.get("/platform-costs/logs", params={"page_size": 201})
    assert response.status_code == 422


def test_get_logs_invalid_page_size_zero() -> None:
    """page_size = 0 (below ge=1) must be rejected with 422."""
    response = client.get("/platform-costs/logs", params={"page_size": 0})
    assert response.status_code == 422


def test_get_logs_invalid_page_negative() -> None:
    """page < 1 must be rejected with 422."""
    response = client.get("/platform-costs/logs", params={"page": 0})
    assert response.status_code == 422


def test_get_dashboard_invalid_date_format() -> None:
    """Malformed start date must be rejected with 422."""
    response = client.get("/platform-costs/dashboard", params={"start": "not-a-date"})
    assert response.status_code == 422


def test_get_dashboard_repeated_requests(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Repeated requests to the dashboard route both return 200."""
    real_dashboard = PlatformCostDashboard(
        by_provider=[],
        by_user=[],
        total_cost_microdollars=42,
        total_requests=1,
        total_users=1,
    )
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_dashboard",
        AsyncMock(return_value=real_dashboard),
    )

    r1 = client.get("/platform-costs/dashboard")
    r2 = client.get("/platform-costs/dashboard")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["total_cost_microdollars"] == 42
    assert r2.json()["total_cost_microdollars"] == 42


def _make_cost_log_row() -> CostLogRow:
    return CostLogRow(
        id="log-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        user_id="user-1",
        email="u***@example.com",
        graph_exec_id="graph-1",
        node_exec_id="node-1",
        block_name="LlmCallBlock",
        provider="anthropic",
        tracking_type="token",
        cost_microdollars=500,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=10,
        cache_creation_tokens=5,
        duration=1.5,
        model="claude-3-5-sonnet-20241022",
    )


def test_export_logs_success(
    mocker: pytest_mock.MockerFixture,
) -> None:
    row = _make_cost_log_row()
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_logs_for_export",
        AsyncMock(return_value=([row], False)),
    )

    response = client.get("/platform-costs/logs/export")
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 1
    assert data["truncated"] is False
    assert len(data["logs"]) == 1
    assert data["logs"][0]["cache_read_tokens"] == 10
    assert data["logs"][0]["cache_creation_tokens"] == 5


def test_export_logs_truncated(
    mocker: pytest_mock.MockerFixture,
) -> None:
    rows = [_make_cost_log_row() for _ in range(3)]
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_logs_for_export",
        AsyncMock(return_value=(rows, True)),
    )

    response = client.get("/platform-costs/logs/export")
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 3
    assert data["truncated"] is True


def test_export_logs_with_filters(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_export = AsyncMock(return_value=([], False))
    mocker.patch(
        "backend.api.features.admin.platform_cost_routes.get_platform_cost_logs_for_export",
        mock_export,
    )

    response = client.get(
        "/platform-costs/logs/export",
        params={
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "block_name": "LlmCallBlock",
            "tracking_type": "token",
        },
    )
    assert response.status_code == 200
    mock_export.assert_called_once()
    call_kwargs = mock_export.call_args.kwargs
    assert call_kwargs["provider"] == "anthropic"
    assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
    assert call_kwargs["block_name"] == "LlmCallBlock"
    assert call_kwargs["tracking_type"] == "token"


def test_export_logs_requires_admin() -> None:
    import fastapi
    from fastapi import HTTPException

    def reject_jwt(request: fastapi.Request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_jwt_payload] = reject_jwt
    try:
        response = client.get("/platform-costs/logs/export")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()
