"""
Shared test utilities for mocking API responses in block tests.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(
        self,
        json_data: Dict[str, Any],
        status: int = 200,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.json_data = json_data
        self.status = status
        self.headers = headers or {}
        self.text = json.dumps(json_data) if json_data else ""

    def json(self) -> Dict[str, Any]:
        return self.json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockRequests:
    """Mock Requests class for testing HTTP operations."""

    def __init__(self):
        self.get = AsyncMock()
        self.post = AsyncMock()
        self.put = AsyncMock()
        self.patch = AsyncMock()
        self.delete = AsyncMock()
        self.call_history = []

    def setup_response(
        self, method: str, response_data: Dict[str, Any], status: int = 200
    ):
        """Setup a mock response for a specific HTTP method."""
        mock_response = MockResponse(response_data, status)
        getattr(self, method).return_value = mock_response
        return mock_response

    def setup_error(self, method: str, error_message: str, status: int = 400):
        """Setup an error response for a specific HTTP method."""
        error_data = {"error": {"message": error_message}}
        return self.setup_response(method, error_data, status)

    def setup_sequence(self, method: str, responses: list):
        """Setup a sequence of responses for pagination testing."""
        mock_responses = [MockResponse(data, status) for data, status in responses]
        getattr(self, method).side_effect = mock_responses
        return mock_responses

    def assert_called_with_headers(self, method: str, expected_headers: Dict[str, str]):
        """Assert that the method was called with specific headers."""
        mock_method = getattr(self, method)
        assert mock_method.called
        actual_headers = mock_method.call_args.kwargs.get("headers", {})
        for key, value in expected_headers.items():
            assert (
                actual_headers.get(key) == value
            ), f"Expected header {key}={value}, got {actual_headers.get(key)}"


def load_mock_response(provider: str, response_file: str) -> Dict[str, Any]:
    """Load a mock response from a JSON file."""
    # test_data is now in the same directory as this file
    base_path = Path(__file__).parent / "test_data" / provider / "responses"
    file_path = base_path / response_file

    if not file_path.exists():
        # Return a default response if file doesn't exist
        return {"error": f"Mock response file not found: {response_file}"}

    with open(file_path, "r") as f:
        return json.load(f)


def create_mock_credentials(provider: str, **kwargs) -> MagicMock:
    """Create mock credentials for testing."""
    mock_creds = MagicMock()

    if "api_key" in kwargs:
        mock_creds.api_key.get_secret_value.return_value = kwargs["api_key"]

    if "oauth_token" in kwargs:
        mock_creds.oauth_token.get_secret_value.return_value = kwargs["oauth_token"]

    return mock_creds


class BlockTestHelper:
    """Helper class for testing blocks."""

    @staticmethod
    async def run_block(block, input_data, credentials=None, **kwargs):
        """Run a block and collect all outputs."""
        outputs = []
        async for output in block.run(input_data, credentials=credentials, **kwargs):
            outputs.append(output)
        return outputs

    @staticmethod
    def assert_output_shape(outputs: list, expected_names: list):
        """Assert that outputs have the expected names and structure."""
        assert len(outputs) == len(
            expected_names
        ), f"Expected {len(expected_names)} outputs, got {len(outputs)}"

        actual_names = [output[0] for output in outputs]
        assert (
            actual_names == expected_names
        ), f"Expected output names {expected_names}, got {actual_names}"

    @staticmethod
    def assert_pagination_calls(mock_requests, method: str, expected_calls: int):
        """Assert that pagination made the expected number of API calls."""
        mock_method = getattr(mock_requests, method)
        assert (
            mock_method.call_count == expected_calls
        ), f"Expected {expected_calls} {method} calls, got {mock_method.call_count}"


# Common test responses for different scenarios
COMMON_ERROR_RESPONSES = {
    "unauthorized": {"error": {"message": "Invalid API key", "code": "UNAUTHORIZED"}},
    "rate_limit": {
        "error": {
            "message": "Rate limit exceeded",
            "code": "RATE_LIMIT_EXCEEDED",
            "retry_after": 60,
        }
    },
    "not_found": {"error": {"message": "Resource not found", "code": "NOT_FOUND"}},
    "server_error": {
        "error": {"message": "Internal server error", "code": "INTERNAL_ERROR"}
    },
    "validation_error": {
        "error": {
            "message": "Invalid request parameters",
            "code": "VALIDATION_ERROR",
            "details": [{"field": "name", "message": "Required field missing"}],
        }
    },
}


def create_paginated_response(
    items: list, page_size: int = 10, cursor_field: str = "offset"
) -> list:
    """Create a list of paginated responses for testing."""
    responses = []
    total_items = len(items)

    for i in range(0, total_items, page_size):
        page_items = items[i : i + page_size]
        has_more = i + page_size < total_items

        response = {"items": page_items, "has_more": has_more}

        if has_more:
            if cursor_field == "offset":
                response[cursor_field] = i + page_size
            elif cursor_field == "next_cursor":
                response[cursor_field] = f"cursor_{i + page_size}"

        responses.append((response, 200))

    return responses
