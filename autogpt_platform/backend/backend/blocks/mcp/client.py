"""
MCP (Model Context Protocol) HTTP client.

Implements the MCP Streamable HTTP transport for listing tools and calling tools
on remote MCP servers. Uses JSON-RPC 2.0 over HTTP POST.

Handles both JSON and SSE (text/event-stream) response formats per the MCP spec.
Each discovered tool includes an integrity_hash that callers can pin and verify
before execution to detect tool-poisoning attacks.

Reference: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
"""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.util.request import Requests

if TYPE_CHECKING:
    from backend.blocks.mcp.security import MCPSecurityContext

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool discovered from a server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    # SHA-256 fingerprint of (name + description + inputSchema), set by list_tools().
    # Pin this at block-configuration time and pass to verify_tool_before_call()
    # at execution time to detect post-deployment mutations.
    integrity_hash: str = ""


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

    Pass a MCPSecurityContext to enable MCPS message signing — outgoing payloads
    are wrapped in ECDSA-signed envelopes and MCPS-signed responses are verified.
    Tool integrity checking via verify_tool_before_call() is always available.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str | None = None,
        security_ctx: "MCPSecurityContext | None" = None,
    ):
        from backend.blocks.mcp.helpers import normalize_mcp_url

        self.server_url = normalize_mcp_url(server_url)
        self.auth_token = auth_token
        self.security_ctx = security_ctx
        self._request_id = 0
        self._session_id: str | None = None
        self._server_public_key: str | None = None

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

        Handles both application/json and text/event-stream responses per the
        MCP Streamable HTTP transport specification. When a security context is
        present, the request is MCPS-signed and the response is verified.
        """
        jsonrpc_payload = self._build_jsonrpc_request(method, params)

        if self.security_ctx is not None:
            wire_payload = self.security_ctx.sign_outgoing(jsonrpc_payload)
            logger.debug("MCPS: signed outgoing %s request", method)
        else:
            wire_payload = jsonrpc_payload

        headers = self._build_headers()

        requests = Requests(
            raise_for_status=True,
            extra_headers=headers,
        )
        response = await requests.post(self.server_url, json=wire_payload)

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

        # Plain JSON-RPC responses (no MCPS metadata) pass through unchanged,
        # so non-MCPS servers work without any modification.
        if self.security_ctx is not None:
            body = self.security_ctx.verify_incoming(
                body, server_public_key=self._server_public_key
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
        Returns the server's capabilities. If the server advertises an MCPS
        public key in capabilities._mcps.public_key it is stored for use in
        response signature verification.
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

        capabilities = (result or {}).get("capabilities", {})
        mcps_meta = capabilities.get("_mcps", {})
        if isinstance(mcps_meta, dict) and "public_key" in mcps_meta:
            self._server_public_key = mcps_meta["public_key"]
            logger.debug("MCPS: captured server public key for '%s'", self.server_url)

        return result or {}

    async def list_tools(self) -> list[MCPTool]:
        """
        Discover available tools from the MCP server.

        Returns a list of MCPTool objects. Each tool's integrity_hash field
        is computed from its definition and can be pinned for later verification.
        """
        from backend.blocks.mcp.security import compute_tool_hash

        result = await self._send_request("tools/list")
        if not result or "tools" not in result:
            return []

        tools = []
        for tool_data in result["tools"]:
            tool_dict = {
                "name": tool_data.get("name", ""),
                "description": tool_data.get("description", ""),
                "inputSchema": tool_data.get("inputSchema", {}),
            }
            tools.append(
                MCPTool(
                    name=tool_dict["name"],
                    description=tool_dict["description"],
                    input_schema=tool_dict["inputSchema"],
                    integrity_hash=compute_tool_hash(tool_dict),
                )
            )
        return tools

    async def verify_tool_before_call(
        self, tool_name: str, expected_hash: str
    ) -> None:
        """Re-fetch the tool list and verify that tool_name's definition hasn't changed.

        Raises MCPToolIntegrityError if the hash doesn't match (tool was mutated).
        Raises MCPClientError if the tool is no longer advertised by the server.
        This makes one extra tools/list request and should only be called when
        integrity checking is enabled in the block configuration.
        """
        from backend.blocks.mcp.security import verify_tool_hash

        result = await self._send_request("tools/list")
        if not result or "tools" not in result:
            raise MCPClientError(
                f"Could not retrieve tool list for integrity check of '{tool_name}'"
            )

        for tool_data in result["tools"]:
            if tool_data.get("name") == tool_name:
                verify_tool_hash(tool_data, expected_hash)
                logger.debug("Tool integrity verified for '%s'", tool_name)
                return

        raise MCPClientError(
            f"Tool '{tool_name}' is no longer advertised by the server — "
            "it may have been removed or renamed after the block was configured."
        )

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
