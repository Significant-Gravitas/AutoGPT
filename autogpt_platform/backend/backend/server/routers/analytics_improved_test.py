"""Example of analytics tests with improved error handling and assertions."""

import json
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest_mock
from pytest_snapshot.plugin import Snapshot

import backend.server.routers.analytics as analytics_routes
from backend.server.conftest import TEST_USER_ID
from backend.server.test_helpers import (
    assert_error_response_structure,
    assert_mock_called_with_partial,
    assert_response_status,
    safe_parse_json,
)
from backend.server.utils import get_user_id

app = fastapi.FastAPI()
app.include_router(analytics_routes.router)

client = fastapi.testclient.TestClient(app)


def override_get_user_id() -> str:
    """Override get_user_id for testing"""
    return TEST_USER_ID


app.dependency_overrides[get_user_id] = override_get_user_id


def test_log_raw_metric_success_improved(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful raw metric logging with improved assertions."""
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

    # Improved assertions with better error messages
    assert_response_status(response, 200, "Metric logging should succeed")
    response_data = safe_parse_json(response, "Metric response parsing")

    assert response_data == "metric-123-uuid", f"Unexpected response: {response_data}"

    # Verify the function was called with correct parameters
    assert_mock_called_with_partial(
        mock_log_metric,
        user_id=TEST_USER_ID,
        metric_name="page_load_time",
        metric_value=2.5,
        data_string="/dashboard",
    )

    # Snapshot test the response
    configured_snapshot.assert_match(
        json.dumps({"metric_id": response_data}, indent=2, sort_keys=True),
        "analytics_log_metric_success_improved",
    )


def test_log_raw_metric_invalid_request_improved() -> None:
    """Test invalid metric request with improved error assertions."""
    # Test missing required fields
    response = client.post("/log_raw_metric", json={})

    error_data = assert_error_response_structure(
        response, expected_status=422, expected_error_fields=["loc", "msg", "type"]
    )

    # Verify specific error details
    detail = error_data["detail"]
    assert isinstance(detail, list), "Error detail should be a list"
    assert len(detail) > 0, "Should have at least one error"

    # Check that required fields are mentioned in errors
    error_fields = [error["loc"][-1] for error in detail if "loc" in error]
    assert "metric_name" in error_fields, "Should report missing metric_name"
    assert "metric_value" in error_fields, "Should report missing metric_value"
    assert "data_string" in error_fields, "Should report missing data_string"


def test_log_raw_metric_type_validation_improved() -> None:
    """Test metric type validation with improved assertions."""
    invalid_requests = [
        {
            "data": {
                "metric_name": "test",
                "metric_value": "not_a_number",  # Invalid type
                "data_string": "test",
            },
            "expected_error": "Input should be a valid number",
        },
        {
            "data": {
                "metric_name": "",  # Empty string
                "metric_value": 1.0,
                "data_string": "test",
            },
            "expected_error": "String should have at least 1 character",
        },
        {
            "data": {
                "metric_name": "test",
                "metric_value": float("inf"),  # Infinity
                "data_string": "test",
            },
            "expected_error": "ensure this value is finite",
        },
    ]

    for test_case in invalid_requests:
        response = client.post("/log_raw_metric", json=test_case["data"])

        error_data = assert_error_response_structure(response, expected_status=422)

        # Check that expected error is in the response
        error_text = json.dumps(error_data)
        assert (
            test_case["expected_error"] in error_text
            or test_case["expected_error"].lower() in error_text.lower()
        ), f"Expected error '{test_case['expected_error']}' not found in: {error_text}"
