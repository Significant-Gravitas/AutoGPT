"""
Generic API testing framework for verifying block API calls against expected patterns.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

from backend.sdk import APIKeyCredentials, OAuth2Credentials


class APICallMatcher:
    """Matches actual API calls against expected patterns."""

    def __init__(self, expected: Dict[str, Any]):
        self.expected = expected
        self.url_pattern = expected.get("url_pattern")
        self.method = expected.get("method", "GET").upper()
        self.headers = expected.get("headers", {})
        self.query_params = expected.get("query_params", {})
        self.body_pattern = expected.get("body", {})
        self.response = expected.get("response", {})
        self.status = expected.get("status", 200)

    def matches_url(self, actual_url: str) -> bool:
        """Check if the actual URL matches the expected pattern."""
        if self.url_pattern is None:
            return False

        if "{" in self.url_pattern:
            # Convert URL pattern to regex
            # Replace {param} with named groups
            pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", self.url_pattern)
            pattern = f"^{pattern}$"
            return bool(re.match(pattern, actual_url))
        return actual_url == self.url_pattern

    def matches_headers(self, actual_headers: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Check if required headers are present."""
        errors = []
        for key, expected_value in self.headers.items():
            if key not in actual_headers:
                errors.append(f"Missing required header: {key}")
            elif expected_value and not self._matches_value(
                actual_headers[key], expected_value
            ):
                errors.append(
                    f"Header {key} mismatch: expected {expected_value}, got {actual_headers[key]}"
                )
        return len(errors) == 0, errors

    def matches_query_params(self, actual_url: str) -> Tuple[bool, List[str]]:
        """Check if query parameters match expected values."""
        parsed = urlparse(actual_url)
        actual_params = parse_qs(parsed.query)
        errors = []

        for key, expected_value in self.query_params.items():
            if key not in actual_params:
                if expected_value is not None:  # None means optional
                    errors.append(f"Missing required query param: {key}")
            elif expected_value and not self._matches_value(
                actual_params[key][0], expected_value
            ):
                errors.append(
                    f"Query param {key} mismatch: expected {expected_value}, got {actual_params[key][0]}"
                )

        return len(errors) == 0, errors

    def matches_body(self, actual_body: Any) -> Tuple[bool, List[str]]:
        """Check if request body matches expected pattern."""
        if not self.body_pattern:
            return True, []

        errors = []
        if isinstance(self.body_pattern, dict) and isinstance(actual_body, dict):
            for key, expected_value in self.body_pattern.items():
                if key not in actual_body:
                    if expected_value is not None:
                        errors.append(f"Missing required body field: {key}")
                elif expected_value and not self._matches_value(
                    actual_body[key], expected_value
                ):
                    errors.append(
                        f"Body field {key} mismatch: expected {expected_value}, got {actual_body[key]}"
                    )

        return len(errors) == 0, errors

    def _matches_value(self, actual: Any, expected: Any) -> bool:
        """Check if a value matches the expected pattern."""
        if (
            isinstance(expected, str)
            and expected.startswith("{{")
            and expected.endswith("}}")
        ):
            # Template variable, any non-empty value is acceptable
            return bool(actual)
        elif (
            isinstance(expected, str)
            and expected.startswith("/")
            and expected.endswith("/")
        ):
            # Regex pattern
            pattern = expected[1:-1]
            return bool(re.match(pattern, str(actual)))
        else:
            return actual == expected


class APITestInterceptor:
    """Intercepts API calls and verifies them against expected patterns."""

    def __init__(self, test_data_path: Path):
        self.test_data_path = test_data_path
        self.api_specs = {}
        self.call_log = []
        self.load_api_specs()

    def load_api_specs(self):
        """Load API specifications for all providers."""
        for provider_file in self.test_data_path.glob("*.json"):
            provider_name = provider_file.stem
            with open(provider_file, "r") as f:
                self.api_specs[provider_name] = json.load(f)

    def create_mock_requests(self, provider: str):
        """Create a mock Requests object that intercepts and validates API calls."""
        mock_requests = MagicMock()

        async def mock_request(method: str, url: str, **kwargs):
            """Mock request that validates against expected patterns."""
            # Log the call
            call_info = {
                "method": method.upper(),
                "url": url,
                "headers": kwargs.get("headers", {}),
                "params": kwargs.get("params", {}),
                "json": kwargs.get("json"),
                "data": kwargs.get("data"),
            }
            self.call_log.append(call_info)

            # Find matching pattern
            provider_spec = self.api_specs.get(provider, {})
            api_calls = provider_spec.get("api_calls", [])

            for expected_call in api_calls:
                matcher = APICallMatcher(expected_call)

                # Check if this call matches
                if matcher.method == method.upper() and matcher.matches_url(url):
                    # Validate the call
                    errors = []

                    # Check headers
                    headers_match, header_errors = matcher.matches_headers(
                        kwargs.get("headers", {})
                    )
                    errors.extend(header_errors)

                    # Check query params
                    if kwargs.get("params"):
                        # Build URL with params for checking
                        from urllib.parse import urlencode

                        param_str = urlencode(kwargs["params"])
                        full_url = f"{url}?{param_str}"
                    else:
                        full_url = url

                    params_match, param_errors = matcher.matches_query_params(full_url)
                    errors.extend(param_errors)

                    # Check body
                    body = kwargs.get("json") or kwargs.get("data")
                    if body:
                        body_match, body_errors = matcher.matches_body(body)
                        errors.extend(body_errors)

                    # If validation fails, raise an error
                    if errors:
                        raise AssertionError(
                            "API call validation failed:\n" + "\n".join(errors)
                        )

                    # Return mock response
                    mock_response = AsyncMock()
                    mock_response.status = matcher.status
                    mock_response.json.return_value = matcher.response
                    mock_response.text = json.dumps(matcher.response)
                    return mock_response

            # No matching pattern found
            raise AssertionError(f"No matching API pattern found for {method} {url}")

        # Set up mock methods
        mock_requests.get = AsyncMock(
            side_effect=lambda url, **kwargs: mock_request("GET", url, **kwargs)
        )
        mock_requests.post = AsyncMock(
            side_effect=lambda url, **kwargs: mock_request("POST", url, **kwargs)
        )
        mock_requests.put = AsyncMock(
            side_effect=lambda url, **kwargs: mock_request("PUT", url, **kwargs)
        )
        mock_requests.patch = AsyncMock(
            side_effect=lambda url, **kwargs: mock_request("PATCH", url, **kwargs)
        )
        mock_requests.delete = AsyncMock(
            side_effect=lambda url, **kwargs: mock_request("DELETE", url, **kwargs)
        )

        return mock_requests

    def get_test_scenarios(
        self, provider: str, block_name: str
    ) -> List[Dict[str, Any]]:
        """Get test scenarios for a specific block."""
        provider_spec = self.api_specs.get(provider, {})
        return provider_spec.get("test_scenarios", {}).get(block_name, [])

    def create_test_credentials(self, provider: str) -> Any:
        """Create test credentials based on provider configuration."""
        provider_spec = self.api_specs.get(provider, {})
        auth_type = provider_spec.get("auth_type", "api_key")

        if auth_type == "api_key":
            from backend.sdk import ProviderName

            return APIKeyCredentials(
                provider=ProviderName(provider),
                api_key=provider_spec.get("test_api_key", "test-key"),
            )
        elif auth_type == "oauth2":
            from backend.sdk import ProviderName

            return OAuth2Credentials(
                provider=ProviderName(provider),
                access_token=provider_spec.get("test_access_token", "test-token"),
                refresh_token=provider_spec.get("test_refresh_token", ""),
                scopes=[],
            )
        elif auth_type == "user_password":
            from backend.sdk import ProviderName, UserPasswordCredentials

            return UserPasswordCredentials(
                provider=ProviderName(provider),
                username=provider_spec.get("test_username", "test-user"),
                password=provider_spec.get("test_password", "test-pass"),
            )
        else:
            raise ValueError(f"Unknown auth type: {auth_type}")

    def clear_log(self):
        """Clear the call log."""
        self.call_log = []

    def get_call_summary(self) -> str:
        """Get a summary of all API calls made."""
        summary = []
        for i, call in enumerate(self.call_log, 1):
            summary.append(f"{i}. {call['method']} {call['url']}")
            if call["params"]:
                summary.append(f"   Params: {call['params']}")
            if call["json"]:
                summary.append(f"   Body: {json.dumps(call['json'], indent=2)}")
        return "\n".join(summary)
