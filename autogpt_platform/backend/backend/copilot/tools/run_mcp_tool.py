"""Tool for discovering and executing MCP (Model Context Protocol) server tools."""

import logging
from typing import Any
from urllib.parse import urlparse

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPClient, MCPClientError
from backend.blocks.mcp.helpers import (
    auto_lookup_mcp_credential,
    invalidate_mcp_credential,
    normalize_mcp_url,
    parse_mcp_content,
    server_host,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.file_ref import (
    FILE_REF_PREFIX,
    FileRefExpansionError,
    expand_file_refs_in_args,
)
from backend.copilot.tools.utils import build_missing_credentials_from_field_info
from backend.util.request import HTTPClientError, validate_url_host

from .base import BaseTool
from .models import (
    ErrorResponse,
    MCPToolInfo,
    MCPToolOutputResponse,
    MCPToolsDiscoveredResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)

logger = logging.getLogger(__name__)

# HTTP status codes that indicate authentication is required
_AUTH_STATUS_CODES = {401, 403}


def _service_name(host: str) -> str:
    """Strip the 'mcp.' prefix from an MCP hostname: 'mcp.sentry.dev' → 'sentry.dev'"""
    return host[4:] if host.startswith("mcp.") else host


def _args_contain_file_ref(value: Any) -> bool:
    """True if any nested string in *value* holds an ``@@agptfile:`` token.

    Cheap pre-check so we only pay the ``list_tools`` round-trip (for
    type-aware expansion) when a reference is actually present.
    """
    if isinstance(value, str):
        return FILE_REF_PREFIX in value
    if isinstance(value, dict):
        return any(_args_contain_file_ref(v) for v in value.values())
    if isinstance(value, list):
        return any(_args_contain_file_ref(item) for item in value)
    return False


class RunMCPToolTool(BaseTool):
    """
    Tool for discovering and executing tools on any MCP server.

    Stage 1 — discovery: call with just server_url to get available tools.
    Stage 2 — execution: call with server_url + tool_name + tool_arguments.
    If the server requires OAuth credentials that the user hasn't connected yet,
    a SetupRequirementsResponse is returned so the frontend can render the
    same OAuth login UI as the graph builder.
    """

    @property
    def name(self) -> str:
        return "run_mcp_tool"

    @property
    def description(self) -> str:
        return (
            "Discover and execute MCP server tools. "
            "Call with server_url only to list tools, then with tool_name + tool_arguments to execute. "
            "Call get_mcp_guide first for server URLs and auth."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "MCP server URL (Streamable HTTP endpoint).",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Tool to execute. Omit to discover available tools.",
                },
                "tool_arguments": {
                    "type": "object",
                    "description": "Arguments matching the tool's input schema.",
                },
                "surface_connect_card": {
                    "type": "boolean",
                    "description": (
                        "When true, return only the sign-in card for this "
                        "server (no MCPClient call). Use for 'connect to "
                        "<service>' intent without an action — the card "
                        "indicates connected/not-connected state and offers "
                        "Reconnect."
                    ),
                },
            },
            "required": ["server_url"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        server_url: str = "",
        tool_name: str = "",
        tool_arguments: dict[str, Any] | None = None,
        surface_connect_card: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        server_url = server_url.strip()
        tool_name = tool_name.strip()
        session_id = session.session_id

        # Session-level dry_run prevents real MCP tool execution.
        # Discovery (no tool_name) is still allowed so the agent can inspect
        # available tools, but actual execution is blocked.
        if session.dry_run and tool_name:
            return MCPToolOutputResponse(
                message=(
                    f"[dry-run] MCP tool '{tool_name}' on "
                    f"{server_host(server_url)} was not executed "
                    "because the session is in dry-run mode."
                ),
                server_url=server_url,
                tool_name=tool_name,
                result=None,
                success=True,
                session_id=session_id,
            )

        if tool_arguments is not None and not isinstance(tool_arguments, dict):
            return ErrorResponse(
                message="tool_arguments must be a JSON object.",
                session_id=session_id,
            )
        resolved_tool_arguments: dict[str, Any] = (
            tool_arguments if isinstance(tool_arguments, dict) else {}
        )

        if not server_url:
            return ErrorResponse(
                message="Please provide a server_url for the MCP server.",
                session_id=session_id,
            )

        _parsed = urlparse(server_url)
        if _parsed.username or _parsed.password:
            return ErrorResponse(
                message=(
                    "Do not include credentials in server_url. "
                    "Use the MCP credential setup flow instead."
                ),
                session_id=session_id,
            )
        if _parsed.query or _parsed.fragment:
            return ErrorResponse(
                message=(
                    "Do not include query parameters or fragments in server_url. "
                    "Use the MCP credential setup flow instead."
                ),
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                session_id=session_id,
            )

        # Validate URL to prevent SSRF — blocks loopback and private IP ranges
        try:
            await validate_url_host(server_url)
        except ValueError as e:
            msg = str(e)
            if "Unable to resolve" in msg or "No IP addresses" in msg:
                user_msg = (
                    f"Hostname not found: {server_host(server_url)}. "
                    "Please check the URL — the domain may not exist."
                )
            else:
                user_msg = f"Blocked server URL: {msg}"
            return ErrorResponse(message=user_msg, session_id=session_id)

        # Fast DB lookup — no network call.
        # Normalize for matching because stored credentials use normalized URLs.
        creds = await auto_lookup_mcp_credential(user_id, normalize_mcp_url(server_url))
        auth_token = creds.access_token.get_secret_value() if creds else None

        # "Just connect" intent: return only the setup card so the user
        # gets a visible Connect/Reconnect affordance even when there's
        # nothing else to render (previously: discovery succeeded
        # silently and nothing rendered, so the model's "click Connect"
        # promise mismatched the empty UI).
        #
        # When stored creds exist we still need to know whether they're
        # *valid* before promising the user "Connected" — a stale row
        # left over from a server-side revocation would mislead the UI
        # into showing the Reconnect pill, then 401 on the next real
        # tool call (see the John bug this PR also fixes).  Cheapest
        # verification is ``MCPClient.initialize`` (one round-trip, no
        # tool listing); on 401/403 we treat the cred as stale, drop the
        # dead row, and surface the not-connected card so the user
        # re-auths in one step.  Other HTTP errors (timeouts, 5xx) are
        # treated as "unknown, optimistically connected" — the next
        # real tool call will self-correct via the same invalidate path.
        if surface_connect_card:
            connected = creds is not None
            if creds is not None:
                probe_client = MCPClient(server_url, auth_token=auth_token)
                try:
                    try:
                        await probe_client.initialize()
                    except HTTPClientError as probe_err:
                        if probe_err.status_code in _AUTH_STATUS_CODES:
                            await invalidate_mcp_credential(user_id, creds.id)
                            connected = False
                        # Other HTTP statuses (5xx, redirects, etc.) →
                        # leave the cred in place and report
                        # "optimistically connected" — the user can
                        # still try; the real tool call will surface
                        # the actual error if it persists.
                    except Exception:
                        # Any non-HTTP failure (asyncio.TimeoutError,
                        # network errors, MCPClientError, etc.) — also
                        # treat as "unknown, optimistically connected".
                        # Important: we MUST NOT let a transient server
                        # outage delete the user's still-valid cred.
                        # Catching ``Exception`` broadly here keeps the
                        # surface_connect_card fast-path resilient
                        # instead of propagating an uncaught exception
                        # out of ``_execute``.
                        logger.debug(
                            "MCP probe for surface_connect_card failed for %s — "
                            "reporting optimistically connected",
                            server_host(server_url),
                            exc_info=True,
                        )
                finally:
                    # Terminate the probe session on the MCP server.
                    # Without DELETE, every surface_connect_card call
                    # leaks a session row server-side; under load this
                    # accumulates (sentry MEDIUM bug-prediction).
                    # ``close`` is best-effort and swallows its own
                    # errors.
                    await probe_client.close()
            return self._build_setup_requirements(
                server_url, session_id, connected=connected
            )

        client = MCPClient(server_url, auth_token=auth_token)

        try:
            await client.initialize()

            if not tool_name:
                # Stage 1: Discover available tools
                return await self._discover_tools(client, server_url, session_id)
            else:
                # Stage 2: Execute the selected tool
                return await self._execute_tool(
                    client,
                    server_url,
                    tool_name,
                    resolved_tool_arguments,
                    session_id,
                    user_id,
                    session,
                )

        except HTTPClientError as e:
            if e.status_code in _AUTH_STATUS_CODES:
                # 401/403 → user needs to (re)authenticate.  Fire the setup
                # card whether or not we have a stored credential row: when
                # `creds` is None the user has never connected, and when it
                # is non-None the stored token has been revoked / expired
                # server-side without us knowing (refresh_if_needed only
                # refreshes when local `access_token_expires_at` says so).
                # If we have a stale row, delete it so the next attempt
                # doesn't loop on the same dead token.
                if creds is not None:
                    await invalidate_mcp_credential(user_id, creds.id)
                return self._build_setup_requirements(server_url, session_id)
            host = server_host(server_url)
            logger.warning("MCP HTTP error for %s: status=%s", host, e.status_code)
            return ErrorResponse(
                message=(f"MCP request to {host} failed with HTTP {e.status_code}."),
                session_id=session_id,
                error=f"HTTP {e.status_code}: {str(e)[:300]}",
            )

        except MCPClientError as e:
            logger.warning("MCP client error for %s: %s", server_host(server_url), e)
            return ErrorResponse(
                message=str(e),
                session_id=session_id,
            )

        except Exception:
            logger.error(
                "Unexpected error calling MCP server %s",
                server_host(server_url),
                exc_info=True,
            )
            return ErrorResponse(
                message="An unexpected error occurred connecting to the MCP server. Please try again.",
                session_id=session_id,
            )

    async def _discover_tools(
        self,
        client: MCPClient,
        server_url: str,
        session_id: str,
    ) -> MCPToolsDiscoveredResponse:
        """List available tools from an already-initialized MCPClient.

        Called when the agent invokes run_mcp_tool with only server_url (no
        tool_name). Returns MCPToolsDiscoveredResponse so the agent can
        inspect tool schemas and choose one to execute in a follow-up call.
        """
        tools = await client.list_tools()
        tool_infos = [
            MCPToolInfo(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in tools
        ]
        host = server_host(server_url)
        return MCPToolsDiscoveredResponse(
            message=(
                f"Discovered {len(tool_infos)} tool(s) on {host}. "
                "Call run_mcp_tool again with tool_name and tool_arguments to execute one."
            ),
            server_url=server_url,
            tools=tool_infos,
            session_id=session_id,
        )

    async def _execute_tool(
        self,
        client: MCPClient,
        server_url: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        session_id: str,
        user_id: str | None,
        session: ChatSession,
    ) -> MCPToolOutputResponse | ErrorResponse:
        """Execute a specific tool on an already-initialized MCPClient.

        Before dispatch, any ``@@agptfile:`` references in *tool_arguments* are
        expanded inline so the external server receives the real file contents
        rather than the literal token. The opaque ``tool_arguments`` object is
        skipped by the SDK-level wrapper expansion (it has no declared
        properties), so it must be expanded here using the tool's own schema —
        mirroring how RunBlockTool expands block inputs.

        Parses the MCP content response into a plain Python value:
        - text items: parsed as JSON when possible, kept as str otherwise
        - image items: kept as {type, data, mimeType} dict for frontend rendering
        - resource items: unwrapped to their resource payload dict
        Single-item responses are unwrapped from the list; multiple items are
        returned as a list; empty content returns None.
        """
        if _args_contain_file_ref(tool_arguments):
            input_schema = await self._lookup_tool_schema(client, tool_name)
            try:
                tool_arguments = await expand_file_refs_in_args(
                    tool_arguments, user_id, session, input_schema=input_schema
                )
            except FileRefExpansionError as exc:
                return ErrorResponse(
                    message=(
                        f"Failed to resolve file reference: {exc}. "
                        "Ensure the file exists before referencing it."
                    ),
                    session_id=session_id,
                )

        result = await client.call_tool(tool_name, tool_arguments)

        if result.is_error:
            error_text = " ".join(
                item.get("text", "")
                for item in result.content
                if item.get("type") == "text"
            )
            return ErrorResponse(
                message=f"MCP tool '{tool_name}' returned an error: {error_text or 'Unknown error'}",
                session_id=session_id,
            )

        result_value = parse_mcp_content(result.content)

        return MCPToolOutputResponse(
            message=f"MCP tool '{tool_name}' executed successfully.",
            server_url=server_url,
            tool_name=tool_name,
            result=result_value,
            success=True,
            session_id=session_id,
        )

    async def _lookup_tool_schema(
        self,
        client: MCPClient,
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Return the named tool's input schema for type-aware ref expansion.

        Returns ``None`` if the tool can't be found or listing fails —
        expansion still proceeds, just without schema-driven type coercion
        (e.g. keeping a string field as raw text instead of parsing JSON).
        """
        try:
            tools = await client.list_tools()
        except Exception as exc:
            # Schema lookup is best-effort — any failure (HTTP 4xx/5xx,
            # timeouts, connection errors, protocol errors) degrades to
            # schema-less expansion rather than failing the tool call.
            logger.debug(
                "Could not list tools on %s for schema lookup: %s",
                server_host(client.server_url),
                exc,
            )
            return None
        return next(
            (t.input_schema for t in tools if t.name == tool_name),
            None,
        )

    def _build_setup_requirements(
        self,
        server_url: str,
        session_id: str,
        connected: bool = False,
    ) -> SetupRequirementsResponse | ErrorResponse:
        """Build a SetupRequirementsResponse for an MCP server credential.

        ``connected=True`` flips the response into the "already signed in"
        shape — frontend renders "Connected to <service> — Reconnect"
        instead of the bare Connect button.  Used by the
        ``surface_connect_card`` path so the user always gets visible
        feedback even when stored creds are still valid.
        """
        mcp_block = MCPToolBlock()
        credentials_fields_info = mcp_block.input_schema.get_credentials_fields_info()

        # Apply the server_url discriminator value so the frontend's CredentialsGroupedView
        # can match the credential to the correct OAuth provider/server.
        for field_info in credentials_fields_info.values():
            if field_info.discriminator == "server_url":
                field_info.discriminator_values.add(server_url)

        missing_creds_dict = build_missing_credentials_from_field_info(
            credentials_fields_info, matched_keys=set()
        )

        if not missing_creds_dict:
            logger.error(
                "No credential requirements found for MCP server %s — "
                "MCPToolBlock may not have credentials configured",
                server_host(server_url),
            )
            return ErrorResponse(
                message=(
                    f"Unable to connect to {_service_name(server_host(server_url))} "
                    "— no credentials configured."
                ),
                session_id=session_id,
            )

        missing_creds_list = list(missing_creds_dict.values())

        host = server_host(server_url)
        service = _service_name(host)
        message = (
            f"You're connected to {service}. Use Reconnect to swap accounts."
            if connected
            else f"To continue, sign in to {service} and approve access."
        )
        return SetupRequirementsResponse(
            message=message,
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=server_url,
                agent_name=service,
                user_readiness=UserReadiness(
                    has_all_credentials=connected,
                    missing_credentials={} if connected else missing_creds_dict,
                    ready_to_run=connected,
                ),
                requirements={
                    # Keep `requirements.credentials` in sync with
                    # `user_readiness.missing_credentials` — when connected,
                    # neither field should advertise a credential need.
                    "credentials": [] if connected else missing_creds_list,
                    "inputs": [],
                    "execution_modes": ["immediate"],
                },
            ),
            graph_id=None,
            graph_version=None,
        )
