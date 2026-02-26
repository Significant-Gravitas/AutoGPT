"""Tool for discovering and executing MCP (Model Context Protocol) server tools."""

import json
import logging
from typing import Any
from urllib.parse import urlparse

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPClient, MCPClientError
from backend.copilot.model import ChatSession
from backend.copilot.tools.utils import build_missing_credentials_from_field_info
from backend.util.request import HTTPClientError

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
            "Connect to an MCP (Model Context Protocol) server to discover and execute its tools. "
            "Call with just `server_url` to see available tools. "
            "Then call again with `server_url`, `tool_name`, and `tool_arguments` to execute. "
            "If the server requires authentication, the user will be prompted to connect it. "
            "Find MCP servers at https://registry.modelcontextprotocol.io/ — hundreds of integrations "
            "including GitHub, Postgres, Slack, filesystem, and more."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": (
                        "URL of the MCP server (Streamable HTTP endpoint), "
                        "e.g. https://mcp.example.com/mcp"
                    ),
                },
                "tool_name": {
                    "type": "string",
                    "description": (
                        "Name of the MCP tool to execute. "
                        "Omit on first call to discover available tools."
                    ),
                },
                "tool_arguments": {
                    "type": "object",
                    "description": (
                        "Arguments to pass to the selected tool. "
                        "Must match the tool's input schema returned during discovery."
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
        **kwargs,
    ) -> ToolResponseBase:
        server_url: str = (kwargs.get("server_url") or "").strip()
        tool_name: str = (kwargs.get("tool_name") or "").strip()
        tool_arguments: dict[str, Any] = kwargs.get("tool_arguments") or {}
        session_id = session.session_id

        if not server_url:
            return ErrorResponse(
                message="Please provide a server_url for the MCP server.",
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                session_id=session_id,
            )

        # Fast DB lookup — no network call
        creds = await MCPToolBlock._auto_lookup_credential(user_id, server_url)
        auth_token = creds.access_token.get_secret_value() if creds else None

        client = MCPClient(server_url, auth_token=auth_token)

        try:
            await client.initialize()

            if not tool_name:
                # Stage 1: Discover available tools
                return await self._discover_tools(client, server_url, session_id)
            else:
                # Stage 2: Execute the selected tool
                return await self._execute_tool(
                    client, server_url, tool_name, tool_arguments, session_id
                )

        except HTTPClientError as e:
            if e.status_code in _AUTH_STATUS_CODES and not creds:
                # Server requires auth and user has no stored credentials
                return self._build_setup_requirements(server_url, session_id)
            logger.warning("MCP HTTP error for %s: %s", server_url, e)
            return ErrorResponse(
                message=f"MCP server returned HTTP {e.status_code}: {e}",
                session_id=session_id,
            )

        except MCPClientError as e:
            logger.warning("MCP client error for %s: %s", server_url, e)
            return ErrorResponse(
                message=str(e),
                session_id=session_id,
            )

        except Exception as e:
            logger.error(
                "Unexpected error calling MCP server %s", server_url, exc_info=True
            )
            return ErrorResponse(
                message=f"Unexpected error: {e}",
                session_id=session_id,
            )

    async def _discover_tools(
        self,
        client: MCPClient,
        server_url: str,
        session_id: str,
    ) -> MCPToolsDiscoveredResponse:
        tools = await client.list_tools()
        tool_infos = [
            MCPToolInfo(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in tools
        ]
        host = _server_host(server_url)
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
    ) -> MCPToolOutputResponse | ErrorResponse:
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

        # Parse content items into a clean Python value
        output_parts = []
        for item in result.content:
            if item.get("type") == "text":
                text = item.get("text", "")
                try:
                    output_parts.append(json.loads(text))
                except (json.JSONDecodeError, ValueError):
                    output_parts.append(text)
            elif item.get("type") == "image":
                output_parts.append(
                    {
                        "type": "image",
                        "data": item.get("data"),
                        "mimeType": item.get("mimeType"),
                    }
                )
            elif item.get("type") == "resource":
                output_parts.append(item.get("resource", {}))

        result_value = (
            output_parts[0] if len(output_parts) == 1 else (output_parts or None)
        )

        return MCPToolOutputResponse(
            message=f"MCP tool '{tool_name}' executed successfully.",
            server_url=server_url,
            tool_name=tool_name,
            result=result_value,
            success=True,
            session_id=session_id,
        )

    def _build_setup_requirements(
        self,
        server_url: str,
        session_id: str,
    ) -> SetupRequirementsResponse | ErrorResponse:
        """Build a SetupRequirementsResponse for a missing MCP server credential."""
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
                server_url,
            )
            return ErrorResponse(
                message=(
                    f"The MCP server at {_server_host(server_url)} requires authentication, "
                    "but no credential configuration was found."
                ),
                session_id=session_id,
            )

        missing_creds_list = list(missing_creds_dict.values())

        host = _server_host(server_url)
        return SetupRequirementsResponse(
            message=(
                f"The MCP server at {host} requires authentication. "
                "Please connect your credentials to continue."
            ),
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=server_url,
                agent_name=f"MCP: {host}",
                user_readiness=UserReadiness(
                    has_all_credentials=False,
                    missing_credentials=missing_creds_dict,
                    ready_to_run=False,
                ),
                requirements={
                    "credentials": missing_creds_list,
                    "inputs": [],
                    "execution_modes": ["immediate"],
                },
            ),
            graph_id=None,
            graph_version=None,
        )


def _server_host(server_url: str) -> str:
    """Extract the hostname from a server URL for display purposes."""
    try:
        return urlparse(server_url).netloc or server_url
    except Exception:
        return server_url
