"""Helper functions for improved test assertions and error handling."""

import json
from typing import Any, Dict, Optional


def assert_response_status(
    response: Any, expected_status: int = 200, error_context: Optional[str] = None
) -> None:
    """Assert response status with helpful error message.

    Args:
        response: The HTTP response object
        expected_status: Expected status code
        error_context: Optional context to include in error message
    """
    if response.status_code != expected_status:
        error_msg = f"Expected status {expected_status}, got {response.status_code}"
        if error_context:
            error_msg = f"{error_context}: {error_msg}"

        # Try to include response body in error
        try:
            body = response.json()
            error_msg += f"\nResponse body: {json.dumps(body, indent=2)}"
        except Exception:
            error_msg += f"\nResponse text: {response.text}"

        raise AssertionError(error_msg)


def safe_parse_json(
    response: Any, error_context: Optional[str] = None
) -> Dict[str, Any]:
    """Safely parse JSON response with error handling.

    Args:
        response: The HTTP response object
        error_context: Optional context for error messages

    Returns:
        Parsed JSON data

    Raises:
        AssertionError: If JSON parsing fails
    """
    try:
        return response.json()
    except Exception as e:
        error_msg = f"Failed to parse JSON response: {e}"
        if error_context:
            error_msg = f"{error_context}: {error_msg}"
        error_msg += f"\nResponse text: {response.text[:500]}"
        raise AssertionError(error_msg)


def assert_error_response_structure(
    response: Any,
    expected_status: int = 422,
    expected_error_fields: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Assert error response has expected structure.

    Args:
        response: The HTTP response object
        expected_status: Expected error status code
        expected_error_fields: List of expected fields in error detail

    Returns:
        Parsed error response
    """
    assert_response_status(response, expected_status, "Error response check")

    error_data = safe_parse_json(response, "Error response parsing")

    # Check basic error structure
    assert "detail" in error_data, f"Missing 'detail' in error response: {error_data}"

    # Check specific error fields if provided
    if expected_error_fields:
        detail = error_data["detail"]
        if isinstance(detail, list):
            # FastAPI validation errors
            for error in detail:
                assert "loc" in error, f"Missing 'loc' in error: {error}"
                assert "msg" in error, f"Missing 'msg' in error: {error}"
                assert "type" in error, f"Missing 'type' in error: {error}"

    return error_data


def assert_mock_called_with_partial(mock_obj: Any, **expected_kwargs: Any) -> None:
    """Assert mock was called with expected kwargs (partial match).

    Args:
        mock_obj: The mock object to check
        **expected_kwargs: Expected keyword arguments
    """
    assert mock_obj.called, f"Mock {mock_obj} was not called"

    actual_kwargs = mock_obj.call_args.kwargs if mock_obj.call_args else {}

    for key, expected_value in expected_kwargs.items():
        assert (
            key in actual_kwargs
        ), f"Missing key '{key}' in mock call. Actual keys: {list(actual_kwargs.keys())}"
        assert (
            actual_kwargs[key] == expected_value
        ), f"Mock called with {key}={actual_kwargs[key]}, expected {expected_value}"
