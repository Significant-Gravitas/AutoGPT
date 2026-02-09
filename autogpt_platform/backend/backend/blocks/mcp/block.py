"""
MCP (Model Context Protocol) Tool Block.

A single dynamic block that can connect to any MCP server, discover available tools,
and execute them. Works like AgentExecutorBlock — the user selects a tool from a
dropdown and the input/output schema adapts dynamically.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.integrations.credentials_store import IntegrationCredentialsStore

from backend.blocks.mcp.client import MCPClient, MCPClientError
from backend.data.block import (
    Block,
    BlockCategory,
    BlockInput,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.data.model import OAuth2Credentials, SchemaField
from backend.util.json import validate_with_jsonschema

logger = logging.getLogger(__name__)


class MCPToolBlock(Block):
    """
    A block that connects to an MCP server, lets the user pick a tool,
    and executes it with dynamic input/output schema.

    The flow:
    1. User provides an MCP server URL (and optional credentials)
    2. Frontend calls the backend to get tool list from that URL
    3. User selects a tool from a dropdown (available_tools)
    4. The block's input schema updates to reflect the selected tool's parameters
    5. On execution, the block calls the MCP server to run the tool
    """

    class Input(BlockSchemaInput):
        server_url: str = SchemaField(
            description="URL of the MCP server (Streamable HTTP endpoint)",
            placeholder="https://mcp.example.com/mcp",
        )
        credential_id: str = SchemaField(
            description="Credential ID from OAuth flow (empty for public servers)",
            default="",
            hidden=True,
        )
        available_tools: dict[str, Any] = SchemaField(
            description="Available tools on the MCP server. "
            "This is populated automatically when a server URL is provided.",
            default={},
            hidden=True,
        )
        selected_tool: str = SchemaField(
            description="The MCP tool to execute",
            placeholder="Select a tool",
            default="",
        )
        tool_input_schema: dict[str, Any] = SchemaField(
            description="JSON Schema for the selected tool's input parameters. "
            "Populated automatically when a tool is selected.",
            default={},
            hidden=True,
        )

        tool_arguments: dict[str, Any] = SchemaField(
            description="Arguments to pass to the selected MCP tool. "
            "The fields here are defined by the tool's input schema.",
            default={},
        )

        @classmethod
        def get_input_schema(cls, data: BlockInput) -> dict[str, Any]:
            """Return the tool's input schema so the builder UI renders dynamic fields."""
            return data.get("tool_input_schema", {})

        @classmethod
        def get_input_defaults(cls, data: BlockInput) -> BlockInput:
            """Return the current tool_arguments as defaults for the dynamic fields."""
            return data.get("tool_arguments", {})

        @classmethod
        def get_missing_input(cls, data: BlockInput) -> set[str]:
            """Check which required tool arguments are missing."""
            required_fields = cls.get_input_schema(data).get("required", [])
            tool_arguments = data.get("tool_arguments", {})
            return set(required_fields) - set(tool_arguments)

        @classmethod
        def get_mismatch_error(cls, data: BlockInput) -> str | None:
            """Validate tool_arguments against the tool's input schema."""
            tool_schema = cls.get_input_schema(data)
            if not tool_schema:
                return None
            tool_arguments = data.get("tool_arguments", {})
            return validate_with_jsonschema(tool_schema, tool_arguments)

    class Output(BlockSchemaOutput):
        result: Any = SchemaField(description="The result returned by the MCP tool")
        error: str = SchemaField(description="Error message if the tool call failed")

    def __init__(self):
        super().__init__(
            id="a0a4b1c2-d3e4-4f56-a7b8-c9d0e1f2a3b4",
            description="Connect to any MCP server and execute its tools. "
            "Provide a server URL, select a tool, and pass arguments dynamically.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=MCPToolBlock.Input,
            output_schema=MCPToolBlock.Output,
            block_type=BlockType.STANDARD,
            test_input={
                "server_url": "https://mcp.example.com/mcp",
                "selected_tool": "get_weather",
                "tool_input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "tool_arguments": {"city": "London"},
            },
            test_output=[
                (
                    "result",
                    {"weather": "sunny", "temperature": 20},
                ),
            ],
            test_mock={
                "_call_mcp_tool": lambda *a, **kw: {
                    "weather": "sunny",
                    "temperature": 20,
                },
            },
        )

    async def _call_mcp_tool(
        self,
        server_url: str,
        tool_name: str,
        arguments: dict[str, Any],
        auth_token: str | None = None,
    ) -> Any:
        """Call a tool on the MCP server. Extracted for easy mocking in tests."""
        client = MCPClient(server_url, auth_token=auth_token)
        await client.initialize()
        result = await client.call_tool(tool_name, arguments)

        if result.is_error:
            error_text = ""
            for item in result.content:
                if item.get("type") == "text":
                    error_text += item.get("text", "")
            raise MCPClientError(
                f"MCP tool '{tool_name}' returned an error: "
                f"{error_text or 'Unknown error'}"
            )

        # Extract text content from the result
        output_parts = []
        for item in result.content:
            if item.get("type") == "text":
                text = item.get("text", "")
                # Try to parse as JSON for structured output
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

        # If single result, unwrap
        if len(output_parts) == 1:
            return output_parts[0]
        return output_parts if output_parts else None

    async def _resolve_auth_token(
        self, credential_id: str, user_id: str, server_url: str = ""
    ) -> str | None:
        """Resolve a Bearer token from a stored credential ID, refreshing if needed.

        Falls back to looking up credentials by server_url when credential_id
        is empty (e.g. when pruneEmptyValues strips it from the saved graph).
        """
        from backend.integrations.credentials_store import IntegrationCredentialsStore
        from backend.integrations.providers import ProviderName

        store = IntegrationCredentialsStore()
        creds = None

        if credential_id:
            creds = await store.get_creds_by_id(user_id, credential_id)

        # Fallback: look up by server_url (same approach as discover-tools)
        if not creds and server_url:
            logger.info(
                f"credential_id not available, looking up credential by server_url"
            )
            try:
                mcp_creds = await store.get_creds_by_provider(
                    user_id, str(ProviderName.MCP)
                )
                best: OAuth2Credentials | None = None
                for c in mcp_creds:
                    if (
                        isinstance(c, OAuth2Credentials)
                        and c.metadata.get("mcp_server_url") == server_url
                    ):
                        if best is None or (
                            (c.access_token_expires_at or 0)
                            > (best.access_token_expires_at or 0)
                        ):
                            best = c
                creds = best
            except Exception:
                logger.debug("Could not look up MCP credentials by server_url", exc_info=True)

        if not creds:
            return None

        if isinstance(creds, OAuth2Credentials):
            # Refresh if token expires within 5 minutes
            if (
                creds.access_token_expires_at
                and creds.access_token_expires_at < int(time.time()) + 300
            ):
                creds = await self._refresh_mcp_oauth(creds, user_id, store)
            return creds.access_token.get_secret_value()
        if hasattr(creds, "api_key") and creds.api_key:
            return creds.api_key.get_secret_value() or None
        return None

    async def _refresh_mcp_oauth(
        self,
        creds: OAuth2Credentials,
        user_id: str,
        store: "IntegrationCredentialsStore",
    ) -> OAuth2Credentials:
        """Refresh MCP OAuth tokens using metadata stored during the OAuth callback."""
        from backend.blocks.mcp.oauth import MCPOAuthHandler

        metadata = creds.metadata or {}
        token_url = metadata.get("mcp_token_url")
        if not token_url:
            logger.warning(
                f"Cannot refresh MCP credential {creds.id}: no token_url in metadata"
            )
            return creds

        handler = MCPOAuthHandler(
            client_id=metadata.get("mcp_client_id", ""),
            client_secret=metadata.get("mcp_client_secret", ""),
            redirect_uri="",  # Not needed for refresh
            authorize_url="",  # Not needed for refresh
            token_url=token_url,
            resource_url=metadata.get("mcp_resource_url"),
        )

        try:
            fresh = await handler.refresh_tokens(creds)
            await store.update_creds(user_id, fresh)
            logger.info(f"Refreshed MCP OAuth credential {creds.id}")
            return fresh
        except Exception:
            logger.exception(f"Failed to refresh MCP OAuth credential {creds.id}")
            return creds

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        if not input_data.server_url:
            yield "error", "MCP server URL is required"
            return

        if not input_data.selected_tool:
            yield "error", "No tool selected. Please select a tool from the dropdown."
            return

        auth_token = await self._resolve_auth_token(
            input_data.credential_id, user_id, server_url=input_data.server_url
        )

        try:
            result = await self._call_mcp_tool(
                server_url=input_data.server_url,
                tool_name=input_data.selected_tool,
                arguments=input_data.tool_arguments,
                auth_token=auth_token,
            )
            yield "result", result
        except MCPClientError as e:
            yield "error", str(e)
        except Exception as e:
            logger.exception(f"MCP tool call failed: {e}")
            yield "error", f"MCP tool call failed: {str(e)}"
