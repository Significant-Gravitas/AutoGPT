"""HTTP client for platform API.

This client handles communication with the AutoGPT Platform API,
for listing blocks, executing them, and managing credentials.
"""

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class PlatformClientError(Exception):
    """Error from platform API."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class PlatformClient:
    """Client for platform.agpt.co API."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        """Initialize the platform client.

        Args:
            base_url: Platform API base URL.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def list_blocks(self) -> list[dict[str, Any]]:
        """List all available blocks from platform API.

        Returns:
            List of block dictionaries with id, name, description, schemas.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/api/v1/blocks"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(url, headers=self._headers()) as resp:
                    if resp.status >= 400:
                        error_text = await resp.text()
                        raise PlatformClientError(
                            f"Platform API error: {error_text}",
                            status_code=resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                raise PlatformClientError(f"Connection error: {e}") from e

    async def execute_block(
        self,
        block_id: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a block via platform API.

        Args:
            block_id: The block ID to execute.
            input_data: Input data matching the block's input schema.

        Returns:
            Execution result with outputs.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/api/v1/blocks/{block_id}/execute"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url, headers=self._headers(), json=input_data
                ) as resp:
                    if resp.status >= 400:
                        error_text = await resp.text()
                        raise PlatformClientError(
                            f"Platform API error: {error_text}",
                            status_code=resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                raise PlatformClientError(f"Connection error: {e}") from e
