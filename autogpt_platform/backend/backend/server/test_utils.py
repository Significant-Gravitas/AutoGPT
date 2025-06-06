"""Common test utilities and constants for server tests."""

from typing import Any, Dict
from unittest.mock import Mock

import pytest

# Test ID constants
TEST_USER_ID = "test-user-id"
ADMIN_USER_ID = "admin-user-id"
TARGET_USER_ID = "target-user-id"

# Common test data constants
FIXED_TIMESTAMP = "2024-01-01T00:00:00Z"
TRANSACTION_UUID = "transaction-123-uuid"
METRIC_UUID = "metric-123-uuid"
ANALYTICS_UUID = "analytics-123-uuid"


def create_mock_with_id(mock_id: str) -> Mock:
    """Create a mock object with an id attribute.

    Args:
        mock_id: The ID value to set on the mock

    Returns:
        Mock object with id attribute set
    """
    return Mock(id=mock_id)


def assert_status_and_parse_json(
    response: Any, expected_status: int = 200
) -> Dict[str, Any]:
    """Assert response status and return parsed JSON.

    Args:
        response: The HTTP response object
        expected_status: Expected status code (default: 200)

    Returns:
        Parsed JSON response data

    Raises:
        AssertionError: If status code doesn't match expected
    """
    assert (
        response.status_code == expected_status
    ), f"Expected status {expected_status}, got {response.status_code}: {response.text}"
    return response.json()


@pytest.mark.parametrize(
    "metric_value,metric_name,data_string",
    [
        (100, "api_calls_count", "external_api"),
        (0, "error_count", "no_errors"),
        (-5.2, "temperature_delta", "cooling"),
        (1.23456789, "precision_test", "float_precision"),
        (999999999, "large_number", "max_value"),
    ],
)
def parametrized_metric_values_decorator(func):
    """Decorator for parametrized metric value tests."""
    return pytest.mark.parametrize(
        "metric_value,metric_name,data_string",
        [
            (100, "api_calls_count", "external_api"),
            (0, "error_count", "no_errors"),
            (-5.2, "temperature_delta", "cooling"),
            (1.23456789, "precision_test", "float_precision"),
            (999999999, "large_number", "max_value"),
        ],
    )(func)
