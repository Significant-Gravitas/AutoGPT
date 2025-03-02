"""
API module for Example API integration.

This module provides a example of how to create a client for an API.
"""

# We also have a Json Wrapper library available in backend.util.json
from json import JSONDecodeError
from typing import Any, Dict, Optional

from backend.data.model import APIKeyCredentials

# This is a wrapper around the requests library that is used to make API requests.
from backend.util.request import Requests


class ExampleAPIException(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _get_headers(credentials: APIKeyCredentials) -> dict[str, str]:
    return {
        "Authorization": credentials.api_key.get_secret_value(),
        "Content-Type": "application/json",
    }


class ExampleClient:
    """Client for the Example API"""

    API_BASE_URL = "https://api.example.com/v1"

    def __init__(
        self,
        credentials: Optional[APIKeyCredentials] = None,
        custom_requests: Optional[Requests] = None,
    ):
        if custom_requests:
            self._requests = custom_requests
        else:
            headers: Dict[str, str] = {
                "Content-Type": "application/json",
            }
            if credentials:
                headers["Authorization"] = credentials.auth_header()

            self._requests = Requests(
                extra_headers=headers,
                trusted_origins=["https://api.example.com"],
                raise_for_status=False,
            )

    def _handle_response(self, response) -> Any:
        """
        Handles API response and checks for errors.

        Args:
            response: The response object from the request.

        Returns:
            The parsed JSON response data.

        Raises:
            ExampleAPIException: If the API request fails.
        """
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "")
            except JSONDecodeError:
                error_message = response.text

            raise ExampleAPIException(
                f"Example API request failed ({response.status_code}): {error_message}",
                response.status_code,
            )

        response_data = response.json()
        if "errors" in response_data:
            error_messages = [
                error.get("message", "") for error in response_data["errors"]
            ]
            raise ExampleAPIException(
                f"Example API returned errors: {', '.join(error_messages)}",
                response.status_code,
            )

        return response_data

    def get_resource(self, resource_id: str) -> Dict:
        """
        Fetches a resource from the Example API.

        Args:
            resource_id: The ID of the resource to fetch.

        Returns:
            The resource data as a dictionary.

        Raises:
            ExampleAPIException: If the API request fails.
        """
        try:
            response = self._requests.get(
                f"{self.API_BASE_URL}/resources/{resource_id}"
            )
            return self._handle_response(response)
        except ExampleAPIException:
            raise
        except Exception as e:
            raise ExampleAPIException(f"Failed to get resource: {str(e)}", 500)

    def create_resource(self, data: Dict) -> Dict:
        """
        Creates a new resource via the Example API.

        Args:
            data: The resource data to create.

        Returns:
            The created resource data as a dictionary.

        Raises:
            ExampleAPIException: If the API request fails.
        """
        try:
            response = self._requests.post(f"{self.API_BASE_URL}/resources", json=data)
            return self._handle_response(response)
        except ExampleAPIException:
            raise
        except Exception as e:
            raise ExampleAPIException(f"Failed to create resource: {str(e)}", 500)
