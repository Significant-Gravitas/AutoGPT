"""
V2 External API - MCP Server Endpoint

Exposes the platform's Copilot tools as an MCP (Model Context Protocol) server,
allowing external MCP clients (Claude Desktop, Cursor, etc.) to interact with
agents, runs, library, and other platform features programmatically.

Uses Streamable HTTP transport with stateless sessions, authenticated via the
same API key / OAuth bearer token mechanism as the rest of the external API.
"""

import logging
from typing import Any, Sequence

import pydantic
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from prisma.enums import APIKeyPermission
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

from backend.copilot.model import ChatSession
from backend.copilot.sdk.tool_adapter import _build_input_schema, _execute_tool_sync
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool
from backend.data.auth.api_key import validate_api_key
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    validate_access_token,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_mcp_server() -> FastMCP:
    """Create the MCP server with all eligible Copilot tools registered."""
    settings = Settings()
    base_url = settings.config.platform_base_url or "https://platform.agpt.co"

    server = FastMCP(
        name="autogpt-platform",
        instructions=(
            "AutoGPT Platform MCP Server. "
            "Use these tools to find, create, run, and manage AI agents."
        ),
        token_verifier=ExternalAPITokenVerifier(),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(base_url),
            resource_server_url=AnyHttpUrl(f"{base_url}/external-api/v2/mcp"),
        ),
        stateless_http=True,
        streamable_http_path="/",
    )

    registered: list[str] = []
    for tool in TOOL_REGISTRY.values():
        allowed, required_perms = tool.allow_external_use
        if not allowed or required_perms is None:
            logger.debug(f"Skipping MCP tool {tool.name} (not allowed externally)")
            continue
        _register_tool(server, tool, required_perms)
        registered.append(tool.name)

    logger.info(f"MCP server created with {len(registered)} tools: {registered}")
    return server


def create_mcp_app() -> Starlette:
    """Create the Starlette ASGI app for the MCP server."""
    server = create_mcp_server()
    return server.streamable_http_app()


# ---------------------------------------------------------------------------
# Token verification — reuses existing external API auth infrastructure
# ---------------------------------------------------------------------------


class ExternalAPITokenVerifier(TokenVerifier):
    """Validates API keys and OAuth tokens via external API auth."""

    async def verify_token(self, token: str) -> AccessToken | None:
        # Try API key first
        api_key_info = await validate_api_key(token)
        if api_key_info:
            return AccessToken(
                token=token,
                client_id=api_key_info.user_id,
                scopes=[s.value for s in api_key_info.scopes],
            )

        # Try OAuth bearer token
        try:
            token_info, _ = await validate_access_token(token)
            return AccessToken(
                token=token,
                client_id=token_info.user_id,
                scopes=[s.value for s in token_info.scopes],
            )
        except (InvalidClientError, InvalidTokenError):
            return None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _create_tool_handler(
    tool: BaseTool,
    required_scopes: Sequence[str],
):
    """Create an async MCP tool handler that wraps a BaseTool subclass.

    The handler checks that the caller's API key / OAuth token
    has all `required_scopes` before executing the tool.
    """

    async def handler(ctx: Context, **kwargs: Any) -> str:
        access_token = get_access_token()
        if not access_token:
            return "Authentication required"

        # Enforce per-tool permission scopes
        if required_scopes:
            missing = [s for s in required_scopes if s not in access_token.scopes]
            if missing:
                return f"Missing required permission(s): " f"{', '.join(missing)}"

        user_id = access_token.client_id
        session = ChatSession.new(user_id)

        result = await _execute_tool_sync(tool, user_id, session, kwargs)

        parts = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts) if parts else ""

    return handler


def _register_tool(
    server: FastMCP, tool: BaseTool, required_perms: Sequence[APIKeyPermission]
) -> None:
    """Register a Copilot tool on the MCP server."""
    required_scopes = [p.value for p in required_perms]
    handler = _create_tool_handler(tool, required_scopes)

    mcp_tool = MCPTool(
        fn=handler,
        name=tool.name,
        title=None,
        description=tool.description,
        parameters=_build_input_schema(tool),
        fn_metadata=_PASSTHROUGH_META,
        is_async=True,
        context_kwarg="ctx",
        annotations=None,
    )
    server._tool_manager._tools[tool.name] = mcp_tool


# ---------------------------------------------------------------------------
# Passthrough arg model — lets us specify JSON Schema directly instead of
# having FastMCP introspect the handler function's signature.
# ---------------------------------------------------------------------------


class _PassthroughArgs(ArgModelBase):
    """Accepts any fields and passes them through as kwargs."""

    model_config = pydantic.ConfigDict(extra="allow")

    def model_dump_one_level(self, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.__pydantic_extra__ or {})


_PASSTHROUGH_META = FuncMetadata(
    arg_model=_PassthroughArgs,
    output_schema=None,
    output_model=None,
    wrap_output=False,
)
