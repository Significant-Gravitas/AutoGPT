"""
MCP (Model Context Protocol) Tool Block.

A single dynamic block that can connect to any MCP server, discover available tools,
and execute them. Works like AgentExecutorBlock — the user selects a tool from a
dropdown and the input/output schema adapts dynamically.
"""

import json
import logging
from typing import Any, Literal

from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.blocks.mcp.client import MCPClient, MCPClientError
from backend.data.block import BlockInput, BlockOutput
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.json import validate_with_jsonschema

logger = logging.getLogger(__name__)

TEST_CREDENTIALS = OAuth2Credentials(
    id="test-mcp-cred",
    provider="mcp",
    access_token=SecretStr("mock-mcp-token"),
    refresh_token=SecretStr("mock-refresh"),
    scopes=[],
    title="Mock MCP credential",
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


MCPCredentials = CredentialsMetaInput[Literal[ProviderName.MCP], Literal["oauth2"]]


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
        credentials: MCPCredentials = CredentialsField(
            discriminator="server_url",
            description="MCP server OAuth credentials",
            default={},
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
            block_type=BlockType.MCP_TOOL,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "server_url": "https://mcp.example.com/mcp",
                "credentials": TEST_CREDENTIALS_INPUT,
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

    @staticmethod
    async def _auto_lookup_credential(
        user_id: str, server_url: str
    ) -> "OAuth2Credentials | None":
        """Auto-lookup stored MCP credential for a server URL.

        This is a fallback for nodes that don't have ``credentials`` explicitly
        set (e.g. nodes created before the credential field was wired up).
        """
        from backend.integrations.creds_manager import IntegrationCredentialsManager
        from backend.integrations.providers import ProviderName

        try:
            mgr = IntegrationCredentialsManager()
            mcp_creds = await mgr.store.get_creds_by_provider(
                user_id, ProviderName.MCP.value
            )
            best: OAuth2Credentials | None = None
            for cred in mcp_creds:
                if (
                    isinstance(cred, OAuth2Credentials)
                    and (cred.metadata or {}).get("mcp_server_url") == server_url
                ):
                    if best is None or (
                        (cred.access_token_expires_at or 0)
                        > (best.access_token_expires_at or 0)
                    ):
                        best = cred
            if best:
                best = await mgr.refresh_if_needed(user_id, best)
                logger.info(
                    "Auto-resolved MCP credential %s for %s", best.id, server_url
                )
            return best
        except Exception:
            logger.warning("Auto-lookup MCP credential failed", exc_info=True)
            return None

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        credentials: OAuth2Credentials | None = None,
        **kwargs,
    ) -> BlockOutput:
        if not input_data.server_url:
            yield "error", "MCP server URL is required"
            return

        if not input_data.selected_tool:
            yield "error", "No tool selected. Please select a tool from the dropdown."
            return

        # Validate required tool arguments before calling the server.
        # The executor-level validation is bypassed for MCP blocks because
        # get_input_defaults() flattens tool_arguments, stripping tool_input_schema
        # from the validation context.
        required = set(input_data.tool_input_schema.get("required", []))
        if required:
            missing = required - set(input_data.tool_arguments.keys())
            if missing:
                yield "error", (
                    f"Missing required argument(s): {', '.join(sorted(missing))}. "
                    f"Please fill in all required fields marked with * in the block form."
                )
                return

        # If no credentials were injected by the executor (e.g. legacy nodes
        # that don't have the credentials field set), try to auto-lookup
        # the stored MCP credential for this server URL.
        if credentials is None:
            credentials = await self._auto_lookup_credential(
                user_id, input_data.server_url
            )

        auth_token = (
            credentials.access_token.get_secret_value() if credentials else None
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
