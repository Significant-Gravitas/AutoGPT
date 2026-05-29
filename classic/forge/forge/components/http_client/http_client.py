import json
import logging
from typing import Any, Iterator, Optional

import requests
from pydantic import BaseModel, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import HTTPError

logger = logging.getLogger(__name__)


class HTTPClientConfiguration(BaseModel):
    default_timeout: int = Field(
        default=30, description="Default timeout in seconds for HTTP requests"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for failed requests"
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="List of allowed domains (empty = all domains allowed)",
    )
    user_agent: str = Field(
        default="AutoGPT-HTTPClient/1.0",
        description="User agent string for requests",
    )
    max_response_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum response size in bytes",
    )


class HTTPClientComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[HTTPClientConfiguration]
):
    """Provides commands to make HTTP requests."""

    config_class = HTTPClientConfiguration

    def __init__(self, config: Optional[HTTPClientConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})

    def get_resources(self) -> Iterator[str]:
        yield "Ability to make HTTP requests to external APIs."

    def get_commands(self) -> Iterator[Command]:
        yield self.http_get
        yield self.http_post
        yield self.http_put
        yield self.http_delete

    def _is_domain_allowed(self, url: str) -> bool:
        """Check if the URL's domain is in the allowed list."""
        if not self.config.allowed_domains:
            return True

        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for allowed in self.config.allowed_domains:
            if domain == allowed.lower() or domain.endswith("." + allowed.lower()):
                return True
        return False

    def _make_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request and return a structured response.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: The URL to request
            headers: Optional headers
            params: Optional query parameters
            body: Optional request body
            timeout: Optional timeout override

        Returns:
            dict: Structured response with status, headers, and body
        """
        if not self._is_domain_allowed(url):
            raise HTTPError(
                f"Domain not in allowed list. Allowed: {self.config.allowed_domains}",
                url=url,
            )

        request_timeout = timeout or self.config.default_timeout
        request_headers = headers or {}

        try:
            if method == "GET":
                response = self.session.get(
                    url, headers=request_headers, params=params, timeout=request_timeout
                )
            elif method == "POST":
                response = self.session.post(
                    url,
                    headers=request_headers,
                    params=params,
                    json=body if isinstance(body, dict) else None,
                    data=body if isinstance(body, str) else None,
                    timeout=request_timeout,
                )
            elif method == "PUT":
                response = self.session.put(
                    url,
                    headers=request_headers,
                    params=params,
                    json=body if isinstance(body, dict) else None,
                    data=body if isinstance(body, str) else None,
                    timeout=request_timeout,
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url, headers=request_headers, params=params, timeout=request_timeout
                )
            else:
                raise HTTPError(f"Unsupported HTTP method: {method}", url=url)

            # Check response size
            content_length = len(response.content)
            if content_length > self.config.max_response_size:
                raise HTTPError(
                    f"Response too large: {content_length} bytes "
                    f"(max: {self.config.max_response_size})",
                    status_code=response.status_code,
                    url=url,
                )

            # Try to parse as JSON, fall back to text
            try:
                response_body = response.json()
            except json.JSONDecodeError:
                response_body = response.text

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_body,
                "url": response.url,
            }

        except requests.exceptions.Timeout:
            raise HTTPError(
                f"Request timed out after {request_timeout} seconds", url=url
            )
        except requests.exceptions.ConnectionError as e:
            raise HTTPError(f"Connection error: {e}", url=url)
        except requests.exceptions.RequestException as e:
            raise HTTPError(f"Request failed: {e}", url=url)

    @command(
        ["http_get", "get_request"],
        "Make an HTTP GET request to retrieve data from a URL.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to fetch",
                required=True,
            ),
            "headers": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Optional HTTP headers as key-value pairs",
                required=False,
            ),
            "params": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Optional query parameters",
                required=False,
            ),
            "timeout": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Timeout in seconds (default: 30)",
                minimum=1,
                maximum=300,
                required=False,
            ),
        },
    )
    def http_get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Make an HTTP GET request.

        Args:
            url: The URL to request
            headers: Optional headers
            params: Optional query parameters
            timeout: Optional timeout

        Returns:
            str: JSON-formatted response
        """
        result = self._make_request("GET", url, headers, params, timeout=timeout)
        return json.dumps(result, indent=2)

    @command(
        ["http_post", "post_request"],
        "Make an HTTP POST request to send data to a URL.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to post to",
                required=True,
            ),
            "body": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="The request body (will be sent as JSON)",
                required=False,
            ),
            "headers": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Optional HTTP headers",
                required=False,
            ),
            "timeout": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Timeout in seconds (default: 30)",
                minimum=1,
                maximum=300,
                required=False,
            ),
        },
    )
    def http_post(
        self,
        url: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Make an HTTP POST request.

        Args:
            url: The URL to request
            body: Request body
            headers: Optional headers
            timeout: Optional timeout

        Returns:
            str: JSON-formatted response
        """
        result = self._make_request("POST", url, headers, body=body, timeout=timeout)
        return json.dumps(result, indent=2)

    @command(
        ["http_put", "put_request"],
        "Make an HTTP PUT request to update data at a URL.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to put to",
                required=True,
            ),
            "body": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="The request body (will be sent as JSON)",
                required=True,
            ),
            "headers": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Optional HTTP headers",
                required=False,
            ),
            "timeout": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Timeout in seconds (default: 30)",
                minimum=1,
                maximum=300,
                required=False,
            ),
        },
    )
    def http_put(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Make an HTTP PUT request.

        Args:
            url: The URL to request
            body: Request body
            headers: Optional headers
            timeout: Optional timeout

        Returns:
            str: JSON-formatted response
        """
        result = self._make_request("PUT", url, headers, body=body, timeout=timeout)
        return json.dumps(result, indent=2)

    @command(
        ["http_delete", "delete_request"],
        "Make an HTTP DELETE request to remove a resource.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to delete",
                required=True,
            ),
            "headers": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Optional HTTP headers",
                required=False,
            ),
            "timeout": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Timeout in seconds (default: 30)",
                minimum=1,
                maximum=300,
                required=False,
            ),
        },
    )
    def http_delete(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Make an HTTP DELETE request.

        Args:
            url: The URL to request
            headers: Optional headers
            timeout: Optional timeout

        Returns:
            str: JSON-formatted response
        """
        result = self._make_request("DELETE", url, headers, timeout=timeout)
        return json.dumps(result, indent=2)
