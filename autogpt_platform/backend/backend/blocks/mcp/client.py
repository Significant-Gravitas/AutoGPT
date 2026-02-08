"""
MCP (Model Context Protocol) HTTP client.

Implements the MCP Streamable HTTP transport for listing tools and calling tools
on remote MCP servers. Uses JSON-RPC 2.0 over HTTP POST.

Reference: https://modelcontextprotocol.io/docs/concepts/transports
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.util.request import Requests

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool discovered from a server."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPCallResult:
    """Result from calling an MCP tool."""

    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False


class MCPClientError(Exception):
    """Raised when an MCP protocol error occurs."""

    pass


class MCPClient:
    """
    Async HTTP client for the MCP Streamable HTTP transport.

    Communicates with MCP servers using JSON-RPC 2.0 over HTTP POST.
    Supports optional Bearer token authentication.
    """

    def __init__(self, server_url: str, auth_token: str | None = None):
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _build_jsonrpc_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params is not None:
            req["params"] = params
        return req

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request to the MCP server and return the result."""
        payload = self._build_jsonrpc_request(method, params)
        headers = self._build_headers()

        requests = Requests(raise_for_status=True, extra_headers=headers)
        response = await requests.post(self.server_url, json=payload)
        body = response.json()

        # Handle JSON-RPC error
        if "error" in body:
            error = body["error"]
            raise MCPClientError(
                f"MCP server error [{error.get('code', '?')}]: "
                f"{error.get('message', 'Unknown error')}"
            )

        return body.get("result")

    async def _send_notification(self, method: str) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        headers = self._build_headers()
        notification = {"jsonrpc": "2.0", "method": method}
        requests = Requests(raise_for_status=False, extra_headers=headers)
        await requests.post(self.server_url, json=notification)

    async def initialize(self) -> dict[str, Any]:
        """
        Send the MCP initialize request.

        This is required by the MCP protocol before any other requests.
        Returns the server's capabilities.
        """
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "AutoGPT-Platform", "version": "1.0.0"},
            },
        )
        # Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized")

        return result or {}

    async def list_tools(self) -> list[MCPTool]:
        """
        Discover available tools from the MCP server.

        Returns a list of MCPTool objects with name, description, and input schema.
        """
        result = await self._send_request("tools/list")
        if not result or "tools" not in result:
            return []

        tools = []
        for tool_data in result["tools"]:
            tools.append(
                MCPTool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                )
            )
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPCallResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.

        Returns:
            MCPCallResult with the tool's response content.
        """
        result = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        if not result:
            return MCPCallResult(is_error=True)

        return MCPCallResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )
