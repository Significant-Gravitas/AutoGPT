"""HTTP client for platform API.

This client handles communication with the AutoGPT Platform API,
for listing blocks, executing them, finding agents, and running agents.
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
    """Client for backend.agpt.co API."""

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
            headers["X-API-Key"] = self.api_key
        return headers

    async def list_blocks(self) -> list[dict[str, Any]]:
        """List all available blocks from platform API.

        Returns:
            List of block dictionaries with id, name, description, schemas.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/external-api/v1/blocks"

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
        url = f"{self.base_url}/external-api/v1/blocks/{block_id}/execute"

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

    async def find_agent(self, query: str) -> dict[str, Any]:
        """Search for agents in the platform marketplace.

        Args:
            query: Search query describing what kind of agent is needed.

        Returns:
            Search results with matching agents.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/external-api/v1/tools/find-agent"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url, headers=self._headers(), json={"query": query}
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

    async def run_agent(
        self,
        agent_slug: str,
        inputs: dict[str, Any] | None = None,
        use_defaults: bool = False,
    ) -> dict[str, Any]:
        """Run an agent from the platform marketplace.

        Args:
            agent_slug: Agent slug (e.g. 'username/agent-name').
            inputs: Input values for the agent.
            use_defaults: Whether to run with default values.

        Returns:
            Execution result or setup requirements if inputs are missing.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = f"{self.base_url}/external-api/v1/tools/run-agent"
        payload: dict[str, Any] = {"username_agent_slug": agent_slug}
        if inputs:
            payload["inputs"] = inputs
        if use_defaults:
            payload["use_defaults"] = True

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

    async def get_execution_results(
        self,
        graph_id: str,
        execution_id: str,
    ) -> dict[str, Any]:
        """Get results from a graph/agent execution.

        Args:
            graph_id: The graph (agent) ID.
            execution_id: The execution ID returned by run_agent.

        Returns:
            Execution results with status and outputs.

        Raises:
            PlatformClientError: If the API request fails.
        """
        url = (
            f"{self.base_url}/external-api/v1/graphs/{graph_id}"
            f"/executions/{execution_id}/results"
        )

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
