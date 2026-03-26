import json
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pytest_snapshot.plugin import Snapshot

from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow

from .rate_limit_admin_routes import router as rate_limit_admin_router

app = fastapi.FastAPI()
app.include_router(rate_limit_admin_router)

client = fastapi.testclient.TestClient(app)

_MOCK_MODULE = "backend.api.features.admin.rate_limit_admin_routes"


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _mock_usage_status(
    daily_used: int = 500_000, weekly_used: int = 3_000_000
) -> CoPilotUsageStatus:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    return CoPilotUsageStatus(
        daily=UsageWindow(
            used=daily_used, limit=2_500_000, resets_at=now + timedelta(hours=6)
        ),
        weekly=UsageWindow(
            used=weekly_used, limit=12_500_000, resets_at=now + timedelta(days=3)
        ),
    )


def test_get_rate_limit(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
    target_user_id: str,
) -> None:
    """Test getting rate limit and usage for a user."""
    mocker.patch(
        f"{_MOCK_MODULE}.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(2_500_000, 12_500_000),
    )
    mocker.patch(
        f"{_MOCK_MODULE}.get_usage_status",
        new_callable=AsyncMock,
        return_value=_mock_usage_status(),
    )

    response = client.get("/admin/rate_limit", params={"user_id": target_user_id})

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == target_user_id
    assert data["daily_token_limit"] == 2_500_000
    assert data["weekly_token_limit"] == 12_500_000
    assert data["daily_tokens_used"] == 500_000
    assert data["weekly_tokens_used"] == 3_000_000

    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        "get_rate_limit",
    )


def test_reset_user_usage_daily_only(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
    target_user_id: str,
) -> None:
    """Test resetting only daily usage (default behaviour)."""
    mock_reset = mocker.patch(
        f"{_MOCK_MODULE}.reset_user_usage",
        new_callable=AsyncMock,
    )
    mocker.patch(
        f"{_MOCK_MODULE}.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(2_500_000, 12_500_000),
    )
    mocker.patch(
        f"{_MOCK_MODULE}.get_usage_status",
        new_callable=AsyncMock,
        return_value=_mock_usage_status(daily_used=0, weekly_used=3_000_000),
    )

    response = client.post(
        "/admin/rate_limit/reset",
        json={"user_id": target_user_id},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["daily_tokens_used"] == 0
    # Weekly is untouched
    assert data["weekly_tokens_used"] == 3_000_000

    mock_reset.assert_awaited_once_with(target_user_id, reset_weekly=False)

    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        "reset_user_usage_daily_only",
    )


def test_reset_user_usage_daily_and_weekly(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
    target_user_id: str,
) -> None:
    """Test resetting both daily and weekly usage."""
    mock_reset = mocker.patch(
        f"{_MOCK_MODULE}.reset_user_usage",
        new_callable=AsyncMock,
    )
    mocker.patch(
        f"{_MOCK_MODULE}.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(2_500_000, 12_500_000),
    )
    mocker.patch(
        f"{_MOCK_MODULE}.get_usage_status",
        new_callable=AsyncMock,
        return_value=_mock_usage_status(daily_used=0, weekly_used=0),
    )

    response = client.post(
        "/admin/rate_limit/reset",
        json={"user_id": target_user_id, "reset_weekly": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["daily_tokens_used"] == 0
    assert data["weekly_tokens_used"] == 0

    mock_reset.assert_awaited_once_with(target_user_id, reset_weekly=True)

    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        "reset_user_usage_daily_and_weekly",
    )


def test_reset_user_usage_redis_failure(
    mocker: pytest_mock.MockerFixture,
    target_user_id: str,
) -> None:
    """Test that Redis failure on reset returns 500."""
    mocker.patch(
        f"{_MOCK_MODULE}.reset_user_usage",
        new_callable=AsyncMock,
        side_effect=Exception("Redis connection refused"),
    )

    response = client.post(
        "/admin/rate_limit/reset",
        json={"user_id": target_user_id},
    )

    assert response.status_code == 500


def test_admin_endpoints_require_admin_role(mock_jwt_user) -> None:
    """Test that rate limit admin endpoints require admin role."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    response = client.get("/admin/rate_limit", params={"user_id": "test"})
    assert response.status_code == 403

    response = client.post(
        "/admin/rate_limit/reset",
        json={"user_id": "test"},
    )
    assert response.status_code == 403
