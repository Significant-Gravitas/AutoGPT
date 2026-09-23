"""Run one capability by id: a block, a platform tool, an MCP server or a
skill.

Dispatches to the existing implementations (``RunBlockTool``,
``RunMCPToolTool``, ``ReadSkillTool``, the tool's own ``_execute``) so
credential resolution, pickers, human review, spend approval, dry-run and
error hints all keep working; this module only adds the id resolution and
the per-kind gates.
"""

import logging
from typing import Any
from urllib.parse import urlsplit

from backend.copilot.capabilities.mcp_review import (
    MCPReviewPayload,
    needs_review,
    open_mcp_review,
)
from backend.copilot.capabilities.models import SKILL_TOOL, CapabilityEntry
from backend.copilot.capabilities.registry import configured_tool, get_registry
from backend.copilot.capabilities.resolve import resolve_entry
from backend.copilot.capabilities.sources.mcp_catalog import setup_hint
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.model import ChatSession
from backend.copilot.permissions import BLOCK_GATE, MCP_GATE
from backend.copilot.tool_display import emit_tool_display_name
from backend.data.activity_event import ActivityEventDraft

from .base import BaseTool
from .capability_gates import gate_denied, gate_denied_error
from .describe_capability import MCP_RUN_PARAMETERS, UNKNOWN_ID_HINT, describe_skill
from .models import (
    CapabilityDetailsResponse,
    ErrorResponse,
    ReviewRequiredResponse,
    ToolResponseBase,
)
from .run_block import RunBlockTool
from .run_mcp_tool import RunMCPToolTool
from .session_registry import resolve_session_entry
from .skills import ReadSkillTool

logger = logging.getLogger(__name__)


class RunCapabilityTool(BaseTool):
    digest_large_output = True

    @property
    def name(self) -> str:
        return "run_capability"

    @property
    def description(self) -> str:
        return (
            "Run a capability by id with its input (blocks: the block's inputs; "
            "MCP servers: {tool, arguments}; platform tools: their parameters; "
            "skills: {}). "
            "Never guess ids: take them from find_capability. An unconnected "
            "capability returns a sign-in card: show it and stop. review_required "
            "means wait for approval, then resume_capability(review_id). "
            "validate_only=true inspects without running."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Capability id from find_capability, or an MCP server URL.",
                },
                "input": {
                    "type": "object",
                    "description": "Input for the capability; {} to see what it needs.",
                },
                "validate_only": {
                    "type": "boolean",
                    "description": "Describe what the call would need without running it.",
                    "default": False,
                },
            },
            "required": ["id", "input"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    def activity_event(
        self, session: ChatSession, result: ToolResponseBase, **kwargs
    ) -> ActivityEventDraft | None:
        entry = resolve_entry(get_registry(), str(kwargs.get("id", "")))
        if entry is None:
            return None
        if entry.kind == "block":
            return RunBlockTool().activity_event(session, result)
        if entry.kind == "tool":
            tool = configured_tool(entry.implementations[0].ref)
            payload = kwargs.get("input") or {}
            return tool.activity_event(session, result, **payload) if tool else None
        return None

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        id: str = "",
        input: dict[str, Any] | None = None,
        validate_only: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if input is not None and not isinstance(input, dict):
            return ErrorResponse(
                message="input must be an object", session_id=session_id
            )
        payload: dict[str, Any] = dict(input or {})
        entry = await resolve_session_entry(user_id, session, id)
        if entry is None and id.strip().lower().startswith("https://"):
            # Open world: a server URL the catalog does not know.  The MCP
            # path validates the host; writes pause for review.
            return await _run_mcp(
                None, id.strip(), user_id, session, payload, validate_only
            )
        if entry is None and id.strip().lower().startswith("http://"):
            # Saying "unknown capability" here sent the model looking for a
            # typo in an id that was simply the wrong scheme.
            return ErrorResponse(
                message=(
                    "MCP servers must be reached over https. Retry with the "
                    "https:// form of that URL, or ask the user for the "
                    "server's secure endpoint."
                ),
                session_id=session_id,
            )
        if entry is None:
            return ErrorResponse(message=UNKNOWN_ID_HINT, session_id=session_id)
        if entry.kind == "block":
            return await _run_block(entry, user_id, session, payload, validate_only)
        if entry.kind == "tool":
            return await _describe_tool(entry, session)
        if entry.kind == "skill":
            return await _run_skill(entry, user_id, session, validate_only)
        if entry.kind == "mcp_server":
            server_url = entry.implementations[0].ref
            if not server_url:
                return ErrorResponse(
                    message=setup_hint(entry.schema_ref or entry.id),
                    session_id=session_id,
                )
            return await _run_mcp(
                entry, server_url, user_id, session, payload, validate_only
            )
        return ErrorResponse(
            message=f"Capabilities of kind '{entry.kind}' cannot run yet.",
            session_id=session_id,
        )


async def _run_block(
    entry: CapabilityEntry,
    user_id: str,
    session: ChatSession,
    payload: dict[str, Any],
    validate_only: bool,
) -> ToolResponseBase:
    if gate_denied(BLOCK_GATE):
        return gate_denied_error("blocks", session.session_id)
    block_id = next(
        (impl.ref for impl in entry.implementations if impl.kind == "block"), ""
    )
    return await RunBlockTool()._execute(
        user_id,
        session,
        block_id=block_id,
        input_data=payload,
        validate_only=validate_only,
    )


async def _describe_tool(
    entry: CapabilityEntry, session: ChatSession
) -> ToolResponseBase:
    """What a platform tool needs, without running it.

    Running one never reaches here: the engines resolve the dispatch into a
    call to the tool itself (``capabilities/dispatch.py``) and run it through
    the one tool path, so its gate, announce, history row and events name it.
    """
    name = entry.implementations[0].ref
    tool = configured_tool(name)
    if tool is None:
        return ErrorResponse(message=UNKNOWN_ID_HINT, session_id=session.session_id)
    if gate_denied(name):
        return gate_denied_error(name, session.session_id)
    return CapabilityDetailsResponse(
        message=f"{tool.description} Call again without validate_only to run.",
        capability=entry.listing(),
        parameters=tool.parameters,
        session_id=session.session_id,
    )


async def _run_skill(
    entry: CapabilityEntry, user_id: str, session: ChatSession, validate_only: bool
) -> ToolResponseBase:
    """Load a skill: running one is reading it.

    The engines resolve the dispatch into the ``read_skill`` call it is
    (``capabilities/dispatch.py``) and run that through the one tool path,
    so this answers ``validate_only`` and any engine that did not.
    """
    if gate_denied(SKILL_TOOL):
        return gate_denied_error(SKILL_TOOL, session.session_id)
    if validate_only:
        return describe_skill(entry, session.session_id)
    return await ReadSkillTool()._execute(
        user_id, session, name=entry.implementations[0].ref
    )


async def _run_mcp(
    entry: CapabilityEntry | None,
    server_url: str,
    user_id: str,
    session: ChatSession,
    payload: dict[str, Any],
    validate_only: bool,
) -> ToolResponseBase:
    if gate_denied(MCP_GATE):
        return gate_denied_error("MCP servers", session.session_id)
    tool_name = str(payload.get("tool") or "").strip()
    arguments = payload.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        return ErrorResponse(
            message="input.arguments must be an object", session_id=session.session_id
        )
    connect = bool(payload.get("connect", False))
    if validate_only:
        return CapabilityDetailsResponse(
            message=(
                "MCP server input shape. Call without validate_only and without a "
                "tool to list the server's tools."
            ),
            capability=(
                entry.listing() if entry else {"id": server_url, "kind": "mcp_server"}
            ),
            parameters=MCP_RUN_PARAMETERS,
            session_id=session.session_id,
        )
    host = urlsplit(server_url).hostname or server_url
    if (
        tool_name
        and not session.dry_run
        and needs_review(tool_name, catalog_server=entry is not None)
    ):
        review = MCPReviewPayload(
            server_url=server_url, tool=tool_name, arguments=dict(arguments or {})
        )
        review_id = await open_mcp_review(
            user_id=user_id,
            session_id=session.session_id,
            host=host,
            payload=review,
            organization_id=session.organization_id,
            team_id=session.team_id,
        )
        return ReviewRequiredResponse(
            message=(
                f"'{tool_name}' on {host} looks like a write to a server outside the "
                "official catalog, so it needs the user's approval. Tell the user; "
                f"after they approve, call resume_capability(review_id='{review_id}')."
            ),
            session_id=session.session_id,
            block_id=entry.id if entry else server_url,
            block_name=f"{host}/{tool_name}",
            review_id=review_id,
            graph_exec_id=f"{COPILOT_SESSION_PREFIX}{session.session_id}",
            input_data=review.model_dump(),
        )
    if tool_name:
        emit_tool_display_name(f"{host}: {tool_name}")
    return await RunMCPToolTool()._execute(
        user_id,
        session,
        server_url=server_url,
        tool_name=tool_name,
        tool_arguments=dict(arguments or {}),
        surface_connect_card=connect,
    )
