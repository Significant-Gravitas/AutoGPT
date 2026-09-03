"""
MCP (Model Context Protocol) HTTP client.

Implements the MCP Streamable HTTP transport for listing tools and calling tools
on remote MCP servers. Uses JSON-RPC 2.0 over HTTP POST.

Handles both JSON and SSE (text/event-stream) response formats per the MCP spec.

The client is *dual-era*: it speaks the stateless ``2026-07-28`` revision and
falls back to the ``initialize``-handshake revisions (``2025-03-26``) for
servers that have not upgraded.  Era detection follows the spec's HTTP
backward-compatibility rules and is cached per server URL so legacy servers do
not pay for a failed modern probe on every connection.

References:
  https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
  https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning
  https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from backend.blocks.mcp.protocol import (
    ERROR_HEADER_MISMATCH,
    ERROR_METHOD_NOT_FOUND,
    ERROR_MISSING_CLIENT_CAPABILITY,
    ERROR_UNSUPPORTED_PROTOCOL_VERSION,
    HEADER_METHOD,
    HEADER_NAME,
    HEADER_PROTOCOL_VERSION,
    HEADER_SESSION_ID,
    LEGACY_PROTOCOL_VERSION,
    META_SERVER_INFO,
    MODERN_PROTOCOL_VERSION,
    RESULT_TYPE_COMPLETE,
    RESULT_TYPE_INPUT_REQUIRED,
    HeaderParam,
    InvalidHeaderAnnotation,
    MCPProtocolEra,
    MCPServerProtocol,
    build_request_meta,
    collect_header_params,
    era_cache,
    extract_header_params,
    is_modern_jsonrpc_error,
    jsonrpc_error_code,
    name_header_for,
    negotiate_version,
)
from backend.util.request import HTTPClientError, HTTPServerError, Requests, Response

logger = logging.getLogger(__name__)

# HTTP statuses on which we must *not* guess the server's era: they say
# something about our credentials or rate, not about the protocol.
_NO_FALLBACK_STATUSES = frozenset({401, 403, 407, 429})
# Bound on ``input_required`` retries when the server only wants its
# ``requestState`` echoed back (no interactive input requested).
_MAX_INPUT_REQUIRED_RETRIES = 3


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


@dataclass
class _Reply:
    """A decoded HTTP reply: status, parsed JSON-RPC body (if any), headers."""

    status: int
    body: dict[str, Any] | None
    response: Response

    @property
    def result(self) -> Any:
        return self.body.get("result") if self.body else None

    @property
    def error_code(self) -> int | None:
        return jsonrpc_error_code(self.body)


class MCPClient:
    """
    Async HTTP client for the MCP Streamable HTTP transport.

    Communicates with MCP servers using JSON-RPC 2.0 over HTTP POST.
    Supports optional Bearer token authentication.

    Speaks both the stateless ``2026-07-28`` protocol and the legacy
    ``initialize``-handshake protocol; the era is detected on first contact
    (see :meth:`initialize`) and remembered per server URL.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str | None = None,
    ):
        from backend.blocks.mcp.helpers import normalize_mcp_url

        self.server_url = normalize_mcp_url(server_url)
        self.auth_token = auth_token
        self._request_id = 0
        self._session_id: str | None = None
        self.era: MCPProtocolEra | None = None
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self._header_params: dict[str, list[HeaderParam]] = {}

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
            headers[HEADER_SESSION_ID] = self._session_id
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

    # ───────────────────────── HTTP plumbing ─────────────────────────

    async def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Response:
        """POST one JSON-RPC message and return the raw reply.

        Status handling is done by the caller: modern era detection needs to
        read the body of a ``400`` before deciding what it means.
        """
        requests = Requests(
            raise_for_status=False,
            retry_max_attempts=1,
            extra_headers=headers,
        )
        return await requests.post(self.server_url, json=payload)

    @staticmethod
    def _raise_for_status(response: Response) -> None:
        """Mirror ``Requests(raise_for_status=True)`` for statuses we don't handle."""
        if response.ok or 300 <= response.status < 400:
            return
        message = (
            f"HTTP {response.status} Error: {response.reason}, "
            f"Body: {response.content.decode(errors='replace')}"
        )
        if 400 <= response.status <= 499:
            raise HTTPClientError(message, response.status)
        if 500 <= response.status <= 599:
            raise HTTPServerError(message, response.status)
        raise Exception(message)

    def _decode_reply(self, response: Response, *, strict: bool) -> _Reply:
        """Decode a reply body as JSON or SSE.

        With ``strict`` a body that isn't a JSON object raises; otherwise it
        yields ``body=None`` so era detection can treat it as "not a modern
        JSON-RPC error".
        """
        content_type = response.headers.get("content-type", "")
        body: Any = None
        try:
            if "text/event-stream" in content_type:
                body = self._parse_sse_response(response.text())
            elif response.content.strip():
                body = response.json()
        except MCPClientError:
            if strict:
                raise
        except Exception as e:
            if strict:
                raise MCPClientError(
                    f"MCP server returned non-JSON response: {e}"
                ) from e
        if strict and not isinstance(body, dict):
            raise MCPClientError(
                f"MCP server returned unexpected JSON type: {type(body).__name__}"
            )
        return _Reply(
            status=response.status,
            body=body if isinstance(body, dict) else None,
            response=response,
        )

    @staticmethod
    def _raise_jsonrpc_error(body: dict[str, Any]) -> None:
        error = body["error"]
        if isinstance(error, dict):
            raise MCPClientError(
                f"MCP server error [{error.get('code', '?')}]: "
                f"{error.get('message', 'Unknown error')}"
            )
        raise MCPClientError(f"MCP server error: {error}")

    # ───────────────────────── legacy era ─────────────────────────

    async def _send_legacy(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Legacy request: session header in, ``Mcp-Session-Id`` captured out."""
        payload = self._build_jsonrpc_request(method, params)
        response = await self._post(payload, self._build_headers())
        self._raise_for_status(response)

        session_id = response.headers.get(HEADER_SESSION_ID)
        if session_id:
            self._session_id = session_id

        reply = self._decode_reply(response, strict=True)
        assert reply.body is not None
        if "error" in reply.body:
            self._raise_jsonrpc_error(reply.body)
        return reply.body.get("result")

    async def _send_notification(self, method: str) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        headers = self._build_headers()
        notification = {"jsonrpc": "2.0", "method": method}
        requests = Requests(
            raise_for_status=False,
            extra_headers=headers,
        )
        await requests.post(self.server_url, json=notification)

    async def _legacy_initialize(self) -> dict[str, Any]:
        result = await self._send_legacy(
            "initialize",
            {
                "protocolVersion": LEGACY_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "AutoGPT-Platform", "version": "1.0.0"},
            },
        )
        # Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized")
        result = result or {}
        self.protocol_version = result.get("protocolVersion") or LEGACY_PROTOCOL_VERSION
        self.server_info = result.get("serverInfo") or {}
        self.server_capabilities = result.get("capabilities") or {}
        return result

    # ───────────────────────── modern era ─────────────────────────

    def _modern_headers(
        self, method: str, params: dict[str, Any] | None, protocol_version: str
    ) -> dict[str, str]:
        headers = self._build_headers()
        headers.pop(HEADER_SESSION_ID, None)
        headers[HEADER_PROTOCOL_VERSION] = protocol_version
        headers[HEADER_METHOD] = method
        name = name_header_for(method, params)
        if name is not None:
            headers[HEADER_NAME] = name
        if method == "tools/call" and params:
            header_params = self._header_params.get(str(params.get("name")))
            if header_params:
                headers.update(
                    extract_header_params(header_params, params.get("arguments"))
                )
        return headers

    async def _post_modern(
        self, method: str, params: dict[str, Any] | None, protocol_version: str
    ) -> _Reply:
        body_params = dict(params or {})
        body_params["_meta"] = build_request_meta(protocol_version)
        payload = self._build_jsonrpc_request(method, body_params)
        headers = self._modern_headers(method, params, protocol_version)
        response = await self._post(payload, headers)
        return self._decode_reply(response, strict=False)

    def _adopt_negotiated_version(self, reply: _Reply) -> str:
        """Handle ``UnsupportedProtocolVersionError``: pick a shared version."""
        error = (reply.body or {}).get("error") or {}
        data = error.get("data") if isinstance(error, dict) else None
        supported = data.get("supported") if isinstance(data, dict) else None
        version = negotiate_version(supported)
        if version is None:
            raise MCPClientError(
                "MCP server does not support any protocol version this client "
                f"speaks (server offers: {supported or 'unknown'})"
            )
        self.protocol_version = version
        return version

    async def _send_modern(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        _retried_version: bool = False,
        _retried_headers: bool = False,
    ) -> Any:
        version = self.protocol_version or MODERN_PROTOCOL_VERSION
        reply = await self._post_modern(method, params, version)

        code = reply.error_code
        if code == ERROR_UNSUPPORTED_PROTOCOL_VERSION and not _retried_version:
            self._adopt_negotiated_version(reply)
            self._remember_era()
            return await self._send_modern(method, params, _retried_version=True)

        if (
            code == ERROR_HEADER_MISMATCH
            and method == "tools/call"
            and params
            and not _retried_headers
        ):
            # The server's tool definition may have gained ``x-mcp-header``
            # annotations since we last listed tools: refresh and retry once.
            await self._refresh_header_params()
            return await self._send_modern(method, params, _retried_headers=True)

        if code == ERROR_MISSING_CLIENT_CAPABILITY:
            raise MCPClientError(
                f"MCP server requires client capabilities this client does not "
                f"provide for {method}: "
                f"{(reply.body or {}).get('error', {}).get('data')}"
            )

        if reply.status == 404 and code == ERROR_METHOD_NOT_FOUND:
            raise MCPClientError(f"MCP server does not support {method}")

        if not reply.response.ok:
            if reply.body and "error" in reply.body:
                self._raise_jsonrpc_error(reply.body)
            self._raise_for_status(reply.response)

        if reply.body is None:
            # 2xx without a usable JSON-RPC body: surface the same errors the
            # strict decoder would have raised.
            self._decode_reply(reply.response, strict=True)
            raise MCPClientError("MCP server returned an empty response")

        if "error" in reply.body:
            self._raise_jsonrpc_error(reply.body)

        result = reply.body.get("result")
        result_type = result.get("resultType") if isinstance(result, dict) else None
        if result_type not in (None, RESULT_TYPE_COMPLETE, RESULT_TYPE_INPUT_REQUIRED):
            raise MCPClientError(
                f"MCP server returned unrecognized resultType {result_type!r}"
            )
        if isinstance(result, dict):
            server_info = (result.get("_meta") or {}).get(META_SERVER_INFO)
            if isinstance(server_info, dict):
                self.server_info = server_info
        return result

    async def _refresh_header_params(self) -> None:
        try:
            await self.list_tools()
        except Exception:
            logger.debug("Could not refresh MCP tool schemas", exc_info=True)

    @staticmethod
    def _discover_result_to_init(
        result: dict[str, Any], version: str
    ) -> dict[str, Any]:
        """Shape a ``server/discover`` result like an ``initialize`` result.

        Callers written against the legacy handshake read ``protocolVersion``
        and ``serverInfo`` from what :meth:`initialize` returns.
        """
        init = dict(result)
        init["protocolVersion"] = version
        init.setdefault("capabilities", {})
        init["serverInfo"] = (result.get("_meta") or {}).get(META_SERVER_INFO) or {}
        return init

    # ───────────────────────── era detection ─────────────────────────

    def _remember_era(self) -> None:
        if self.era is not None and self.protocol_version:
            era_cache.set(
                self.server_url, MCPServerProtocol(self.era, self.protocol_version)
            )

    async def _probe_modern(self) -> dict[str, Any] | None:
        """Send ``server/discover`` and classify the server's era.

        Returns the discover result when the server is modern, ``None`` when
        it is legacy.  Raises for auth/rate/5xx statuses because those say
        nothing about the era and callers already handle them.
        """
        version = MODERN_PROTOCOL_VERSION
        reply = await self._post_modern("server/discover", None, version)

        if reply.error_code == ERROR_UNSUPPORTED_PROTOCOL_VERSION:
            version = self._adopt_negotiated_version(reply)
            reply = await self._post_modern("server/discover", None, version)
            if reply.error_code == ERROR_UNSUPPORTED_PROTOCOL_VERSION:
                raise MCPClientError(
                    "MCP server rejected the protocol version it advertised"
                )

        if reply.status in _NO_FALLBACK_STATUSES or reply.status >= 500:
            self._raise_for_status(reply.response)

        if reply.response.ok:
            result = reply.result
            if isinstance(result, dict) and (
                "supportedVersions" in result or "resultType" in result
            ):
                self.protocol_version = version
                return result
            if is_modern_jsonrpc_error(reply.body):
                self._raise_jsonrpc_error(reply.body or {})
            # A legacy server answering an unknown method with a JSON-RPC
            # error (``-32601``) or something else entirely.
            return None

        # 4xx: modern servers use 400 for header/version/capability errors and
        # 404 + -32601 for unknown methods.  server/discover is mandatory on
        # modern servers, so a -32601 means legacy.
        if is_modern_jsonrpc_error(reply.body):
            self._raise_jsonrpc_error(reply.body or {})
        return None

    async def initialize(self) -> dict[str, Any]:
        """
        Establish contact with the server using whichever protocol era it speaks.

        Modern servers (``2026-07-28``) are stateless: this sends
        ``server/discover`` and records the negotiated version.  Legacy servers
        get the ``initialize`` / ``notifications/initialized`` handshake.

        Returns an ``initialize``-shaped dict (``protocolVersion``,
        ``serverInfo``, ``capabilities``) in both cases.
        """
        if self.era is None:
            cached = era_cache.get(self.server_url)
            if cached is not None:
                self.era = cached.era
                self.protocol_version = cached.protocol_version

        if self.era is None:
            result = await self._probe_modern()
            if result is not None:
                self.era = MCPProtocolEra.MODERN
                self._remember_era()
                return self._finish_modern_init(result)
            self.era = MCPProtocolEra.LEGACY
            self.protocol_version = None
            result = await self._legacy_initialize()
            self._remember_era()
            return result

        if self.era is MCPProtocolEra.MODERN:
            try:
                result = await self._send_modern("server/discover")
            except MCPClientError:
                # Cached assumption failed (e.g. server rolled back): re-probe.
                era_cache.forget(self.server_url)
                self.era = None
                self.protocol_version = None
                return await self.initialize()
            return self._finish_modern_init(result if isinstance(result, dict) else {})

        try:
            result = await self._legacy_initialize()
        except HTTPClientError as e:
            if e.status_code in _NO_FALLBACK_STATUSES:
                raise
            # Cached as legacy but the handshake was rejected: the server may
            # have upgraded to modern-only.  Re-probe once from scratch.
            era_cache.forget(self.server_url)
            self.era = None
            self.protocol_version = None
            self._session_id = None
            return await self.initialize()
        self._remember_era()
        return result

    def _finish_modern_init(self, result: dict[str, Any]) -> dict[str, Any]:
        version = self.protocol_version or MODERN_PROTOCOL_VERSION
        self.protocol_version = version
        self.server_capabilities = result.get("capabilities") or {}
        server_info = (result.get("_meta") or {}).get(META_SERVER_INFO)
        if isinstance(server_info, dict):
            self.server_info = server_info
        return self._discover_result_to_init(result, version)

    # ───────────────────────── public API ─────────────────────────

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request to the MCP server and return the result.

        Handles both ``application/json`` and ``text/event-stream`` responses
        as required by the MCP Streamable HTTP transport specification.
        Detects the server's protocol era on first use.
        """
        if self.era is None:
            await self.initialize()
        if self.era is MCPProtocolEra.MODERN:
            return await self._send_modern(method, params)
        return await self._send_legacy(method, params)

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

    async def close(self) -> None:
        """Release server-side state, if the protocol era has any.

        Legacy servers may have minted an ``Mcp-Session-Id``; the transport
        spec asks clients to ``DELETE`` the session URL so the server can free
        it immediately instead of waiting for a timeout sweep.  Modern servers
        are stateless, so there is nothing to release.  Best-effort: failures
        are swallowed so a flaky close can't poison the caller's flow.
        """
        if not self._session_id:
            return
        try:
            headers = self._build_headers()
            requests = Requests(raise_for_status=False, extra_headers=headers)
            await requests.delete(self.server_url)
        except Exception:
            pass
        finally:
            self._session_id = None

    async def list_tools(self) -> list[MCPTool]:
        """
        Discover available tools from the MCP server.

        Returns a list of MCPTool objects with name, description, and input schema.
        On modern servers, tools whose ``x-mcp-header`` annotations are invalid
        are dropped (and logged), as the transport spec requires.
        """
        result = await self._send_request("tools/list")
        if not result or "tools" not in result:
            return []

        tools = []
        for tool_data in result["tools"]:
            name = tool_data.get("name", "")
            input_schema = tool_data.get("inputSchema", {})
            if self.era is MCPProtocolEra.MODERN:
                try:
                    self._header_params[name] = collect_header_params(input_schema)
                except InvalidHeaderAnnotation as e:
                    logger.warning(
                        "Dropping MCP tool %r from %s: invalid x-mcp-header (%s)",
                        name,
                        self.server_url,
                        e,
                    )
                    self._header_params.pop(name, None)
                    continue
            tools.append(
                MCPTool(
                    name=name,
                    description=tool_data.get("description", ""),
                    input_schema=input_schema,
                )
            )
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        input_schema: dict[str, Any] | None = None,
    ) -> MCPCallResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.
            input_schema: The tool's input schema, if the caller has it.  Lets
                the client mirror ``x-mcp-header`` parameters into headers
                without an extra ``tools/list`` round-trip.

        Returns:
            MCPCallResult with the tool's response content.
        """
        if input_schema is not None and tool_name not in self._header_params:
            try:
                self._header_params[tool_name] = collect_header_params(input_schema)
            except InvalidHeaderAnnotation:
                self._header_params[tool_name] = []

        params: dict[str, Any] = {"name": tool_name, "arguments": arguments}
        result = await self._send_request("tools/call", params)

        # Multi Round-Trip Requests: a modern server may hand back an interim
        # result.  We declare no elicitation/sampling/roots capability, so a
        # compliant server only ever asks us to echo ``requestState``.
        for _ in range(_MAX_INPUT_REQUIRED_RETRIES):
            if not (
                isinstance(result, dict)
                and result.get("resultType") == RESULT_TYPE_INPUT_REQUIRED
            ):
                break
            if result.get("inputRequests"):
                raise MCPClientError(
                    f"MCP tool '{tool_name}' requires interactive input "
                    "(elicitation/sampling) mid-execution, which is not "
                    "supported yet"
                )
            retry_params = dict(params)
            if "requestState" in result:
                retry_params["requestState"] = result["requestState"]
            result = await self._send_request("tools/call", retry_params)
        else:
            raise MCPClientError(
                f"MCP tool '{tool_name}' kept requesting more input after "
                f"{_MAX_INPUT_REQUIRED_RETRIES} attempts"
            )

        if not result:
            return MCPCallResult(is_error=True)

        return MCPCallResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )
