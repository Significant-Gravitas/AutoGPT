"""Spraay x402 Gateway API client wrapper."""

import json
import logging
from typing import Any

import requests

from ._config import SPRAAY_GATEWAY_BASE_URL

logger = logging.getLogger(__name__)


class SpraayAPIError(Exception):
    """Raised when a Spraay gateway API call fails."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Spraay API error {status_code}: {message}")


def spraay_request(
    method: str,
    endpoint: str,
    api_key: str,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Make a request to the Spraay x402 gateway.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path (e.g. "/v1/batch/send")
        api_key: Spraay API key for authentication
        json_body: JSON request body (for POST/PUT)
        params: Query parameters (for GET)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response as a dictionary

    Raises:
        SpraayAPIError: If the API returns an error status
    """
    url = f"{SPRAAY_GATEWAY_BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=timeout,
        )

        if response.status_code >= 400:
            error_msg = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get(
                    "error", error_data.get("message", str(error_data))
                )
            except (json.JSONDecodeError, ValueError):
                pass
            raise SpraayAPIError(response.status_code, error_msg)

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            raise SpraayAPIError(
                response.status_code,
                f"Invalid JSON in response: {response.text[:200]}",
            )

    except requests.RequestException as e:
        raise SpraayAPIError(0, f"Connection error: {str(e)}")
