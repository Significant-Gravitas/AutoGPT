"""HTTP client for platform API - used for block execution.

This client handles communication with the AutoGPT Platform API,
which manages credentials and executes blocks with proper authentication.
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
    """Client for platform.agpt.co API - used for block execution."""

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

    async def execute_block(
        self,
        block_id: str,
        input_data: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        """Execute a block via platform API.

        Args:
            block_id: The block ID to execute.
            input_data: Input data matching the block's input schema.
            user_id: User ID for credential resolution.

        Returns:
            Execution result with outputs.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/api/v1/blocks/{block_id}/execute"
        payload = {"input_data": input_data, "user_id": user_id}

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url, headers=self._headers(), json=payload
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

    async def check_credentials(
        self,
        block_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Check if user has required credentials for a block.

        Args:
            block_id: The block ID to check.
            user_id: User ID for credential lookup.

        Returns:
            Credential check result with has_required_credentials and missing list.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/api/v1/blocks/{block_id}/credentials/check"
        params = {"user_id": user_id}

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(
                    url, headers=self._headers(), params=params
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

    async def get_block_info(self, block_id: str) -> dict[str, Any]:
        """Get block information from platform API.

        Args:
            block_id: The block ID to get info for.

        Returns:
            Block information including schema and description.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/api/v1/blocks/{block_id}"

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
