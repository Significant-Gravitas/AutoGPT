"""Tests for analytics API endpoints."""

import json
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from pytest_snapshot.plugin import Snapshot

from .analytics import router as analytics_router

app = fastapi.FastAPI()
app.include_router(analytics_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module."""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


# =============================================================================
# /log_raw_metric endpoint tests
# =============================================================================


def test_log_raw_metric_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test successful raw metric logging."""
    mock_result = Mock(id="metric-123-uuid")
    mock_log_metric = mocker.patch(
        "backend.data.analytics.log_raw_metric",
        new_callable=AsyncMock,
        return_value=mock_result,
    )

    request_data = {
        "metric_name": "page_load_time",
        "metric_value": 2.5,
        "data_string": "/dashboard",
    }

    response = client.post("/log_raw_metric", json=request_data)

    assert response.status_code == 200, f"Unexpected response: {response.text}"
    assert response.json() == "metric-123-uuid"

    mock_log_metric.assert_called_once_with(
        user_id=test_user_id,
        metric_name="page_load_time",
        metric_value=2.5,
        data_string="/dashboard",
    )

    configured_snapshot.assert_match(
        json.dumps({"metric_id": response.json()}, indent=2, sort_keys=True),
        "analytics_log_metric_success",
    )


@pytest.mark.parametrize(
    "metric_value,metric_name,data_string,test_id",
    [
        (100, "api_calls_count", "external_api", "integer_value"),
        (0, "error_count", "no_errors", "zero_value"),
        (-5.2, "temperature_delta", "cooling", "negative_value"),
        (1.23456789, "precision_test", "float_precision", "float_precision"),
        (999999999, "large_number", "max_value", "large_number"),
        (0.0000001, "tiny_number", "min_value", "tiny_number"),
    ],
)
def test_log_raw_metric_various_values(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    metric_value: float,
    metric_name: str,
    data_string: str,
    test_id: str,
) -> None:
    """Test raw metric logging with various metric values."""
    mock_result = Mock(id=f"metric-{test_id}-uuid")
    mocker.patch(
        "backend.data.analytics.log_raw_metric",
        new_callable=AsyncMock,
        return_value=mock_result,
    )

    request_data = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "data_string": data_string,
    }

    response = client.post("/log_raw_metric", json=request_data)

    assert response.status_code == 200, f"Failed for {test_id}: {response.text}"

    configured_snapshot.assert_match(
        json.dumps(
            {"metric_id": response.json(), "test_case": test_id},
            indent=2,
            sort_keys=True,
        ),
        f"analytics_metric_{test_id}",
    )


@pytest.mark.parametrize(
    "invalid_data,expected_error",
    [
        ({}, "Field required"),
        ({"metric_name": "test"}, "Field required"),
        (
            {"metric_name": "test", "metric_value": "not_a_number", "data_string": "x"},
            "Input should be a valid number",
        ),
        (
            {"metric_name": "", "metric_value": 1.0, "data_string": "test"},
            "String should have at least 1 character",
        ),
        (
            {"metric_name": "test", "metric_value": 1.0, "data_string": ""},
            "String should have at least 1 character",
        ),
    ],
    ids=[
        "empty_request",
        "missing_metric_value_and_data_string",
        "invalid_metric_value_type",
        "empty_metric_name",
        "empty_data_string",
    ],
)
def test_log_raw_metric_validation_errors(
    invalid_data: dict,
    expected_error: str,
) -> None:
    """Test validation errors for invalid metric requests."""
    response = client.post("/log_raw_metric", json=invalid_data)

    assert response.status_code == 422
    error_detail = response.json()
    assert "detail" in error_detail, f"Missing 'detail' in error: {error_detail}"

    error_text = json.dumps(error_detail)
    assert (
        expected_error in error_text
    ), f"Expected '{expected_error}' in error response: {error_text}"


def test_log_raw_metric_service_error(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Test error handling when analytics service fails."""
    mocker.patch(
        "backend.data.analytics.log_raw_metric",
        new_callable=AsyncMock,
        side_effect=Exception("Database connection failed"),
    )

    request_data = {
        "metric_name": "test_metric",
        "metric_value": 1.0,
        "data_string": "test",
    }

    response = client.post("/log_raw_metric", json=request_data)

    assert response.status_code == 500
    error_detail = response.json()["detail"]
    assert "Database connection failed" in error_detail["message"]
    assert "hint" in error_detail


# =============================================================================
# /log_raw_analytics endpoint tests
# =============================================================================


def test_log_raw_analytics_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test successful raw analytics logging."""
    mock_result = Mock(id="analytics-789-uuid")
    mock_log_analytics = mocker.patch(
        "backend.data.analytics.log_raw_analytics",
        new_callable=AsyncMock,
        return_value=mock_result,
    )

    request_data = {
        "type": "user_action",
        "data": {
            "action": "button_click",
            "button_id": "submit_form",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {"form_type": "registration", "fields_filled": 5},
        },
        "data_index": "button_click_submit_form",
    }

    response = client.post("/log_raw_analytics", json=request_data)

    assert response.status_code == 200, f"Unexpected response: {response.text}"
    assert response.json() == "analytics-789-uuid"

    mock_log_analytics.assert_called_once_with(
        test_user_id,
        "user_action",
        request_data["data"],
        "button_click_submit_form",
    )

    configured_snapshot.assert_match(
        json.dumps({"analytics_id": response.json()}, indent=2, sort_keys=True),
        "analytics_log_analytics_success",
    )


def test_log_raw_analytics_complex_data(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test raw analytics logging with complex nested data structures."""
    mock_result = Mock(id="analytics-complex-uuid")
    mocker.patch(
        "backend.data.analytics.log_raw_analytics",
        new_callable=AsyncMock,
        return_value=mock_result,
    )

    request_data = {
        "type": "agent_execution",
        "data": {
            "agent_id": "agent_123",
            "execution_id": "exec_456",
            "status": "completed",
            "duration_ms": 3500,
            "nodes_executed": 15,
            "blocks_used": [
                {"block_id": "llm_block", "count": 3},
                {"block_id": "http_block", "count": 5},
                {"block_id": "code_block", "count": 2},
            ],
            "errors": [],
            "metadata": {
                "trigger": "manual",
                "user_tier": "premium",
                "environment": "production",
            },
        },
        "data_index": "agent_123_exec_456",
    }

    response = client.post("/log_raw_analytics", json=request_data)

    assert response.status_code == 200

    configured_snapshot.assert_match(
        json.dumps(
            {"analytics_id": response.json(), "logged_data": request_data["data"]},
            indent=2,
            sort_keys=True,
        ),
        "analytics_log_analytics_complex_data",
    )


@pytest.mark.parametrize(
    "invalid_data,expected_error",
    [
        ({}, "Field required"),
        ({"type": "test"}, "Field required"),
        (
            {"type": "test", "data": "not_a_dict", "data_index": "test"},
            "Input should be a valid dictionary",
        ),
        ({"type": "test", "data": {"key": "value"}}, "Field required"),
    ],
    ids=[
        "empty_request",
        "missing_data_and_data_index",
        "invalid_data_type",
        "missing_data_index",
    ],
)
def test_log_raw_analytics_validation_errors(
    invalid_data: dict,
    expected_error: str,
) -> None:
    """Test validation errors for invalid analytics requests."""
    response = client.post("/log_raw_analytics", json=invalid_data)

    assert response.status_code == 422
    error_detail = response.json()
    assert "detail" in error_detail, f"Missing 'detail' in error: {error_detail}"

    error_text = json.dumps(error_detail)
    assert (
        expected_error in error_text
    ), f"Expected '{expected_error}' in error response: {error_text}"


def test_log_raw_analytics_service_error(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Test error handling when analytics service fails."""
    mocker.patch(
        "backend.data.analytics.log_raw_analytics",
        new_callable=AsyncMock,
        side_effect=Exception("Analytics DB unreachable"),
    )

    request_data = {
        "type": "test_event",
        "data": {"key": "value"},
        "data_index": "test_index",
    }

    response = client.post("/log_raw_analytics", json=request_data)

    assert response.status_code == 500
    error_detail = response.json()["detail"]
    assert "Analytics DB unreachable" in error_detail["message"]
    assert "hint" in error_detail
