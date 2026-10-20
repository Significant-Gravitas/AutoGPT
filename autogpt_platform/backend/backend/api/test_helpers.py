"""Helper functions for improved test assertions and error handling."""

import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


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


@contextmanager
def override_config(settings: Any, attribute: str, value: Any) -> Iterator[None]:
    """Temporarily override a config attribute for testing.

    Warning: Directly mutates settings.config. If config is reloaded or cached
    elsewhere during the test, side effects may leak. Use with caution in
    parallel tests or when config is accessed globally.

    Args:
        settings: The settings object containing .config
        attribute: The config attribute name to override
        value: The temporary value to set
    """
    original = getattr(settings.config, attribute)
    setattr(settings.config, attribute, value)
    try:
        yield
    finally:
        setattr(settings.config, attribute, original)
