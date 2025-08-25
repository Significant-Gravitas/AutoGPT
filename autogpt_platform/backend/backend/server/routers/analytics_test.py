import json
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from pytest_snapshot.plugin import Snapshot

import backend.server.routers.analytics as analytics_routes

app = fastapi.FastAPI()
app.include_router(analytics_routes.router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_log_raw_metric_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test successful raw metric logging"""

    # Mock the analytics function
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

    assert response.status_code == 200
    response_data = response.json()
    assert response_data == "metric-123-uuid"

    # Verify the function was called with correct parameters
    mock_log_metric.assert_called_once_with(
        user_id=test_user_id,
        metric_name="page_load_time",
        metric_value=2.5,
        data_string="/dashboard",
    )

    # Snapshot test the response
    configured_snapshot.assert_match(
        json.dumps({"metric_id": response.json()}, indent=2, sort_keys=True),
        "analytics_log_metric_success",
    )


def test_log_raw_metric_various_values(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test raw metric logging with various metric values"""

    # Mock the analytics function
    mock_result = Mock(id="metric-456-uuid")

    mocker.patch(
        "backend.data.analytics.log_raw_metric",
        new_callable=AsyncMock,
        return_value=mock_result,
    )

    # Test with integer value
    request_data = {
        "metric_name": "api_calls_count",
        "metric_value": 100,
        "data_string": "external_api",
    }

    response = client.post("/log_raw_metric", json=request_data)
    assert response.status_code == 200

    # Test with zero value
    request_data = {
        "metric_name": "error_count",
        "metric_value": 0,
        "data_string": "no_errors",
    }

    response = client.post("/log_raw_metric", json=request_data)
    assert response.status_code == 200

    # Test with negative value
    request_data = {
        "metric_name": "temperature_delta",
        "metric_value": -5.2,
        "data_string": "cooling",
    }

    response = client.post("/log_raw_metric", json=request_data)
    assert response.status_code == 200

    # Snapshot the last response
    configured_snapshot.assert_match(
        json.dumps({"metric_id": response.json()}, indent=2, sort_keys=True),
        "analytics_log_metric_various_values",
    )


def test_log_raw_analytics_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test successful raw analytics logging"""

    # Mock the analytics function
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
            "metadata": {
                "form_type": "registration",
                "fields_filled": 5,
            },
        },
        "data_index": "button_click_submit_form",
    }

    response = client.post("/log_raw_analytics", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data == "analytics-789-uuid"

    # Verify the function was called with correct parameters
    mock_log_analytics.assert_called_once_with(
        test_user_id,
        "user_action",
        request_data["data"],
        "button_click_submit_form",
    )

    # Snapshot test the response
    configured_snapshot.assert_match(
        json.dumps({"analytics_id": response_data}, indent=2, sort_keys=True),
        "analytics_log_analytics_success",
    )


def test_log_raw_analytics_complex_data(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test raw analytics logging with complex nested data"""

    # Mock the analytics function
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
    response_data = response.json()

    # Snapshot test the complex data structure
    configured_snapshot.assert_match(
        json.dumps(
            {
                "analytics_id": response_data,
                "logged_data": request_data["data"],
            },
            indent=2,
            sort_keys=True,
        ),
        "analytics_log_analytics_complex_data",
    )


def test_log_raw_metric_invalid_request() -> None:
    """Test raw metric logging with invalid request data"""
    # Missing required fields
    response = client.post("/log_raw_metric", json={})
    assert response.status_code == 422

    # Invalid metric_value type
    response = client.post(
        "/log_raw_metric",
        json={
            "metric_name": "test",
            "metric_value": "not_a_number",
            "data_string": "test",
        },
    )
    assert response.status_code == 422

    # Missing data_string
    response = client.post(
        "/log_raw_metric",
        json={
            "metric_name": "test",
            "metric_value": 1.0,
        },
    )
    assert response.status_code == 422


def test_log_raw_analytics_invalid_request() -> None:
    """Test raw analytics logging with invalid request data"""
    # Missing required fields
    response = client.post("/log_raw_analytics", json={})
    assert response.status_code == 422

    # Invalid data type (should be dict)
    response = client.post(
        "/log_raw_analytics",
        json={
            "type": "test",
            "data": "not_a_dict",
            "data_index": "test",
        },
    )
    assert response.status_code == 422

    # Missing data_index
    response = client.post(
        "/log_raw_analytics",
        json={
            "type": "test",
            "data": {"key": "value"},
        },
    )
    assert response.status_code == 422
