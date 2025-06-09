"""Example of parametrized tests for analytics endpoints."""

import json
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from pytest_snapshot.plugin import Snapshot

import backend.server.routers.analytics as analytics_routes
from backend.server.conftest import TEST_USER_ID
from backend.server.utils import get_user_id

app = fastapi.FastAPI()
app.include_router(analytics_routes.router)

client = fastapi.testclient.TestClient(app)


def override_get_user_id() -> str:
    """Override get_user_id for testing"""
    return TEST_USER_ID


app.dependency_overrides[get_user_id] = override_get_user_id


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
def test_log_raw_metric_values_parametrized(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    metric_value: float,
    metric_name: str,
    data_string: str,
    test_id: str,
) -> None:
    """Test raw metric logging with various metric values using parametrize."""
    # Mock the analytics function
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

    # Better error handling
    assert response.status_code == 200, f"Failed for {test_id}: {response.text}"
    response_data = response.json()

    # Snapshot test the response
    configured_snapshot.assert_match(
        json.dumps(
            {"metric_id": response_data, "test_case": test_id}, indent=2, sort_keys=True
        ),
        f"analytics_metric_{test_id}",
    )


@pytest.mark.parametrize(
    "invalid_data,expected_error",
    [
        ({}, "Field required"),  # Missing all fields
        ({"metric_name": "test"}, "Field required"),  # Missing metric_value
        (
            {"metric_name": "test", "metric_value": "not_a_number"},
            "Input should be a valid number",
        ),  # Invalid type
        (
            {"metric_name": "", "metric_value": 1.0, "data_string": "test"},
            "String should have at least 1 character",
        ),  # Empty name
    ],
)
def test_log_raw_metric_invalid_requests_parametrized(
    invalid_data: dict,
    expected_error: str,
) -> None:
    """Test invalid metric requests with parametrize."""
    response = client.post("/log_raw_metric", json=invalid_data)

    assert response.status_code == 422
    error_detail = response.json()
    assert "detail" in error_detail
    # Verify error message contains expected error
    error_text = json.dumps(error_detail)
    assert expected_error in error_text or expected_error.lower() in error_text.lower()
