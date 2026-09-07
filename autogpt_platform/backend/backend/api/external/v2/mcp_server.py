"""
V2 External API - MCP Server Endpoint

Exposes the platform's Copilot tools as an MCP (Model Context Protocol) server,
allowing external MCP clients (Claude Desktop, Cursor, etc.) to interact with
agents, runs, library, and other platform features programmatically.

Uses Streamable HTTP transport with stateless sessions, authenticated via the
same API key / OAuth bearer token mechanism as the rest of the external API.

Tool visibility is scope-aware: ``tools/list`` only returns tools whose
required permissions are satisfied by the caller's API key / OAuth token.
"""

import logging
from typing import Any, Sequence

import pydantic
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.server import Context
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.shared.auth import ProtectedResourceMetadata
from prisma.enums import APIKeyPermission
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

from backend.api.external.v2.tenancy import resolve_credential_tenancy
from backend.copilot.model import ChatSession
from backend.copilot.sdk.tool_adapter import _build_input_schema, _execute_tool_sync
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool
from backend.data.auth.api_key import validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    validate_access_token,
)
from backend.util.exceptions import NotAuthorizedError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

META_KEY_REQUIRED_SCOPES = "required_scopes"

# Registry tools deliberately NOT exposed over MCP, with the reason. Every tool
# must either opt in via `BaseTool.allow_external_use` or be listed here;
# mcp_server_test.py fails on anything unclassified so new tools can't slip
# out of the external surface unnoticed.
EXTERNAL_USE_EXCLUSIONS: dict[str, str] = {
    # Copilot-UI or chat-session-only behaviour
    "TodoWrite": "drives the copilot task checklist UI",
    "add_understanding": "copilot's own memory of the user",
    "ask_question": "needs an interactive chat turn to answer",
    "decompose_goal": "renders a plan card in the copilot UI",
    "enter_agent_building_mode": "switches the chat session's mode",
    "schedule_followup": "schedules a copilot chat follow-up, not an agent",
    "run_sub_session": "copilot sub-agents live inside a chat session",
    "get_sub_session_result": "copilot sub-agents live inside a chat session",
    "browser_navigate": "browser state is bound to the chat session",
    "browser_act": "browser state is bound to the chat session",
    "browser_screenshot": "browser state is bound to the chat session",
    # Expert team flows: preview in one turn, confirm in the next; MCP calls
    # get a fresh session each time, and no expert API-key permission exists
    "hire_expert": "two-step confirm needs chat-session state",
    "raise_expert": "two-step confirm needs chat-session state",
    "update_expert": "two-step confirm needs chat-session state",
    "confirm_expert_change": "two-step confirm needs chat-session state",
    "update_expert_soul": "two-step confirm needs chat-session state",
    "confirm_expert_soul_update": "two-step confirm needs chat-session state",
    "handoff_to_expert": "hands off between experts within a chat session",
    "delegate_to_expert": "no API-key permission for expert teams yet",
    "list_team": "no API-key permission for expert teams yet",
    # Copilot-internal knowledge stores (feature-flagged / prompt-facing)
    "memory_store": "copilot Graphiti memory",
    "memory_search": "copilot Graphiti memory",
    "memory_forget_search": "copilot Graphiti memory",
    "memory_forget_confirm": "copilot Graphiti memory",
    "store_skill": "copilot skill registry feeds copilot prompts",
    "read_skill": "copilot skill registry feeds copilot prompts",
    "list_skills": "copilot skill registry feeds copilot prompts",
    "delete_skill": "copilot skill registry feeds copilot prompts",
    # Capabilities the v2 REST API also leaves out, by decision
    "run_block": "direct block execution is not part of v2",
    "continue_run_block": "direct block execution is not part of v2",
    "connect_integration": "OAuth flows are web-UI only in v2",
    "run_mcp_tool": "proxies the user's third-party MCP servers; no permission model",
    "post_to_chat_platform": "posts as the platform bot; no permission model yet",
    "list_chat_platform_channels": "posts as the platform bot; no permission model yet",
    # Security
    "bash_exec": "sandboxed shell on platform infrastructure",
}

# Tools exposed without any permission, with the reason. A tool that spends
# platform money or acts through a platform-owned account must name a permission
# in its `allow_external_use` instead; mcp_server_test.py fails on any other tool
# that opts in with an empty permission list.
UNSCOPED_EXTERNAL_TOOLS: dict[str, str] = {
    "get_doc_page": "serves a published documentation page",
    "search_docs": "searches published documentation",
    "get_mcp_guide": "returns static guidance text",
    "get_agent_building_guide": "returns static guidance text",
    "validate_agent_graph": "pure check over JSON the caller supplied",
    "fix_agent_graph": "pure rewrite of JSON the caller supplied",
    "find_agent": "public marketplace listings, as v2's own /marketplace serves",
}


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


# The path `/mcp` ends up on, counting from the origin: the external API is
# mounted at /external-api, v2 inside it, the MCP app inside that.
MCP_RESOURCE_PATH = "/external-api/v2/mcp"

# RFC 9728 §3.1: the well-known segment is inserted between host and resource
# path, so this is the URL a client derives — and it is on the origin root.
WELL_KNOWN_PROTECTED_RESOURCE_PATH = (
    f"/.well-known/oauth-protected-resource{MCP_RESOURCE_PATH}"
)


def create_mcp_server() -> FastMCP:
    """Create the MCP server with all eligible Copilot tools registered."""
    base_url = _platform_base_url()

    tools = []
    for tool in TOOL_REGISTRY.values():
        allowed, required_perms = tool.allow_external_use
        if not allowed or required_perms is None:
            reason = EXTERNAL_USE_EXCLUSIONS.get(tool.name, "unclassified")
            logger.debug(f"Skipping MCP tool {tool.name}: {reason}")
            continue
        tools.append(_mcp_tool(tool, required_perms))

    logger.info(
        f"MCP server created with {len(tools)} tools: {[t.name for t in tools]}"
    )
    return _ScopeAwareMCP(
        name="autogpt-platform",
        instructions=(
            "AutoGPT Platform MCP Server. "
            "Use these tools to find, create, run, and manage AI agents."
        ),
        token_verifier=ExternalAPITokenVerifier(),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(base_url),
            resource_server_url=AnyHttpUrl(f"{base_url}{MCP_RESOURCE_PATH}"),
        ),
        tools=tools,
        stateless_http=True,
        streamable_http_path="/",
    )


def create_mcp_app() -> Starlette:
    """Create the Starlette ASGI app for the MCP server."""
    server = create_mcp_server()
    return server.streamable_http_app()


def protected_resource_metadata() -> ProtectedResourceMetadata:
    """RFC 9728 metadata for the MCP server, to be served at the ORIGIN root.

    FastMCP registers this document inside the `/mcp` sub-app, so it lands at
    `/external-api/v2/mcp/.well-known/oauth-protected-resource/external-api/v2/mcp`
    while a client — following the `resource_metadata` FastMCP itself puts in
    `WWW-Authenticate` — asks the origin root for it and gets a 404. The root
    app serves this instead; see `WELL_KNOWN_PROTECTED_RESOURCE_PATH`.
    """
    base_url = _platform_base_url()
    return ProtectedResourceMetadata(
        resource=AnyHttpUrl(f"{base_url}{MCP_RESOURCE_PATH}"),
        authorization_servers=[AnyHttpUrl(base_url)],
        resource_name="AutoGPT Platform MCP Server",
    )


def _platform_base_url() -> str:
    return Settings().config.platform_base_url or "https://platform.agpt.co"


# ---------------------------------------------------------------------------
# Scope-aware FastMCP subclass
# ---------------------------------------------------------------------------


class _ScopeAwareMCP(FastMCP):
    """FastMCP subclass that filters ``tools/list`` by the caller's scopes."""

    async def list_tools(self) -> list:
        all_tools = await super().list_tools()
        access_token = get_access_token()
        if not access_token:
            return []
        caller_scopes = set(access_token.scopes)
        return [
            t
            for t in all_tools
            if caller_scopes.issuperset(
                (t.meta or {}).get(META_KEY_REQUIRED_SCOPES, [])
            )
        ]


# ---------------------------------------------------------------------------
# Token verification — reuses existing external API auth infrastructure
# ---------------------------------------------------------------------------


class TenantedAccessToken(AccessToken):
    """An MCP access token carrying the tenant its credentials act in.

    MCP has no per-request dependency chain, so the tenant is resolved here,
    once per token verification, instead of by `v2.tenancy`.
    """

    organization_id: str
    team_id: str | None = None


class ExternalAPITokenVerifier(TokenVerifier):
    """Validates API keys and OAuth tokens via external API auth."""

    async def verify_token(self, token: str) -> AccessToken | None:
        # Try API key first
        api_key_info = await validate_api_key(token)
        if api_key_info:
            return await self._tenanted(token, api_key_info)

        # Try OAuth bearer token
        try:
            token_info, _ = await validate_access_token(token)
            return await self._tenanted(token, token_info)
        except (InvalidClientError, InvalidTokenError):
            return None

    async def _tenanted(
        self, token: str, auth: APIAuthorizationInfo
    ) -> TenantedAccessToken | None:
        try:
            organization_id, team_id = await resolve_credential_tenancy(auth)
        except NotAuthorizedError as e:
            logger.warning(f"Rejecting MCP token for {auth.user_id}: {e}")
            return None
        return TenantedAccessToken(
            token=token,
            client_id=auth.user_id,
            scopes=[s.value for s in auth.scopes],
            organization_id=organization_id,
            team_id=team_id,
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _create_tool_handler(
    tool: BaseTool,
    required_scopes: Sequence[str],
):
    """Create an async MCP tool handler that wraps a BaseTool subclass.

    The handler checks that the caller's API key / OAuth token
    has all ``required_scopes`` before executing the tool.
    """

    async def handler(ctx: Context, **kwargs: Any) -> str:
        # Raised, not returned: a rejection returned as content is reported to
        # the client as a successful call whose text happens to say "denied".
        access_token = get_access_token()
        if not access_token:
            raise ToolError("Authentication required")

        if required_scopes:
            missing = [s for s in required_scopes if s not in access_token.scopes]
            if missing:
                raise ToolError(f"Missing required permission(s): {', '.join(missing)}")

        user_id = access_token.client_id
        organization_id, team_id = (
            (access_token.organization_id, access_token.team_id)
            if isinstance(access_token, TenantedAccessToken)
            else (None, None)
        )
        session = ChatSession.new(
            user_id,
            dry_run=False,
            organization_id=organization_id,
            team_id=team_id,
        )

        result = await _execute_tool_sync(tool, user_id, session, kwargs)

        parts = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts) if parts else ""

    return handler


def _mcp_tool(tool: BaseTool, required_perms: Sequence[APIKeyPermission]) -> MCPTool:
    """Build an MCP tool from a Copilot tool.

    Constructed directly rather than through `FastMCP.add_tool`, which derives
    the schema from the handler's signature; ours is `**kwargs` and the real
    schema comes from the Copilot tool.
    """
    required_scopes = [p.value for p in required_perms]
    return MCPTool(
        fn=_create_tool_handler(tool, required_scopes),
        name=tool.name,
        title=None,
        description=tool.description,
        parameters=_build_input_schema(tool),
        fn_metadata=_PASSTHROUGH_META,
        is_async=True,
        context_kwarg="ctx",
        annotations=None,
        meta={META_KEY_REQUIRED_SCOPES: required_scopes},
    )


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
