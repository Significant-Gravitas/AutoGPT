"""
MCP (Model Context Protocol) HTTP client.

Implements the MCP Streamable HTTP transport for listing tools and calling tools
on remote MCP servers. Uses JSON-RPC 2.0 over HTTP POST.

Handles both JSON and SSE (text/event-stream) response formats per the MCP spec.

Reference: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
"""

import json
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

    def __init__(
        self,
        server_url: str,
        auth_token: str | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self._request_id = 0
        self._session_id: str | None = None

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
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
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

    @staticmethod
    def _parse_sse_response(text: str) -> dict[str, Any]:
        """Parse an SSE (text/event-stream) response body into JSON-RPC data.

        MCP servers may return responses as SSE with format:
            event: message
            data: {"jsonrpc":"2.0","result":{...},"id":1}

        We extract the last `data:` line that contains a JSON-RPC response
        (i.e. has an "id" field), which is the reply to our request.
        """
        last_data: dict[str, Any] | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                payload = stripped[len("data:") :].strip()
                if not payload:
                    continue
                try:
                    parsed = json.loads(payload)
                    # Only keep JSON-RPC responses (have "id"), skip notifications
                    if isinstance(parsed, dict) and "id" in parsed:
                        last_data = parsed
                except (json.JSONDecodeError, ValueError):
                    continue
        if last_data is None:
            raise MCPClientError("No JSON-RPC response found in SSE stream")
        return last_data

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request to the MCP server and return the result.

        Handles both ``application/json`` and ``text/event-stream`` responses
        as required by the MCP Streamable HTTP transport specification.
        """
        payload = self._build_jsonrpc_request(method, params)
        headers = self._build_headers()

        requests = Requests(
            raise_for_status=True,
            extra_headers=headers,
        )
        response = await requests.post(self.server_url, json=payload)

        # Capture session ID from response (MCP Streamable HTTP transport)
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            body = self._parse_sse_response(response.text())
        else:
            try:
                body = response.json()
            except Exception as e:
                raise MCPClientError(
                    f"MCP server returned non-JSON response: {e}"
                ) from e

        if not isinstance(body, dict):
            raise MCPClientError(
                f"MCP server returned unexpected JSON type: {type(body).__name__}"
            )

        # Handle JSON-RPC error
        if "error" in body:
            error = body["error"]
            if isinstance(error, dict):
                raise MCPClientError(
                    f"MCP server error [{error.get('code', '?')}]: "
                    f"{error.get('message', 'Unknown error')}"
                )
            raise MCPClientError(f"MCP server error: {error}")

        return body.get("result")

    async def _send_notification(self, method: str) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        headers = self._build_headers()
        notification = {"jsonrpc": "2.0", "method": method}
        requests = Requests(
            raise_for_status=False,
            extra_headers=headers,
        )
        await requests.post(self.server_url, json=notification)

    async def discover_auth(self) -> dict[str, Any] | None:
        """Probe the MCP server's OAuth metadata (RFC 9728 / MCP spec).

        Returns ``None`` if the server doesn't require auth, otherwise returns
        a dict with:
          - ``authorization_servers``: list of authorization server URLs
          - ``resource``: the resource indicator URL (usually the MCP endpoint)
          - ``scopes_supported``: optional list of supported scopes

        The caller can then fetch the authorization server metadata to get
        ``authorization_endpoint``, ``token_endpoint``, etc.
        """
        from urllib.parse import urlparse

        parsed = urlparse(self.server_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # Build candidates for protected-resource metadata (per RFC 9728)
        path = parsed.path.rstrip("/")
        candidates = []
        if path and path != "/":
            candidates.append(f"{base}/.well-known/oauth-protected-resource{path}")
        candidates.append(f"{base}/.well-known/oauth-protected-resource")

        requests = Requests(
            raise_for_status=False,
        )
        for url in candidates:
            try:
                resp = await requests.get(url)
                if resp.status == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "authorization_servers" in data:
                        return data
            except Exception:
                continue

        return None

    async def discover_auth_server_metadata(
        self, auth_server_url: str
    ) -> dict[str, Any] | None:
        """Fetch the OAuth Authorization Server Metadata (RFC 8414).

        Given an authorization server URL, returns a dict with:
          - ``authorization_endpoint``
          - ``token_endpoint``
          - ``registration_endpoint`` (for dynamic client registration)
          - ``scopes_supported``
          - ``code_challenge_methods_supported``
          - etc.
        """
        from urllib.parse import urlparse

        parsed = urlparse(auth_server_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/")

        # Try standard metadata endpoints (RFC 8414 and OpenID Connect)
        candidates = []
        if path and path != "/":
            candidates.append(f"{base}/.well-known/oauth-authorization-server{path}")
        candidates.append(f"{base}/.well-known/oauth-authorization-server")
        candidates.append(f"{base}/.well-known/openid-configuration")

        requests = Requests(
            raise_for_status=False,
        )
        for url in candidates:
            try:
                resp = await requests.get(url)
                if resp.status == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "authorization_endpoint" in data:
                        return data
            except Exception:
                continue

        return None

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
