"""Full input/output schema for one capability, fetched on demand so the
search index never carries schemas."""

import logging
from typing import Any

from backend.copilot.capabilities.models import SKILL_TOOL, CapabilityEntry
from backend.copilot.capabilities.registry import configured_tool
from backend.copilot.capabilities.schema_trim import collapse_large_enums
from backend.copilot.capabilities.sources.mcp_catalog import setup_hint
from backend.copilot.model import ChatSession
from backend.copilot.permissions import BLOCK_GATE, MCP_GATE

from .base import BaseTool
from .capability_gates import gate_denied, gate_denied_error
from .models import (
    BlockDetailsResponse,
    CapabilityDetailsResponse,
    ErrorResponse,
    MCPToolsDiscoveredResponse,
    ToolResponseBase,
)
from .run_block import RunBlockTool
from .run_mcp_tool import RunMCPToolTool
from .session_registry import resolve_session_entry

logger = logging.getLogger(__name__)

UNKNOWN_ID_HINT = (
    "Unknown capability id. Use the exact 'id' from a find_capability result "
    "(tool:<name>, block:<uuid>, mcp:<host>, skill:<name>)."
)

# A skill takes no input: running it is loading it.
NO_INPUT: dict[str, Any] = {"type": "object", "properties": {}}

# Input shape for run_capability on an MCP server entry.
MCP_RUN_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "description": "Tool name from the server's list."},
        "arguments": {"type": "object", "description": "Arguments for that tool."},
        "connect": {
            "type": "boolean",
            "description": (
                "Only surface the sign-in card for this server (no call). Use "
                "for 'connect to X' with no action."
            ),
        },
    },
}


class DescribeCapabilityTool(BaseTool):
    @property
    def name(self) -> str:
        return "describe_capability"

    @property
    def description(self) -> str:
        return (
            "Inputs and outputs of one capability by id, before its first use. "
            "Blocks return their schema; MCP servers list their tools; platform "
            "tools return their parameters. Large enums are sampled unless "
            "expand=true."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Capability id from find_capability.",
                },
                "expand": {
                    "type": "boolean",
                    "description": "Return every enum value instead of a sample.",
                    "default": False,
                },
            },
            "required": ["id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        id: str = "",
        expand: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        entry = await resolve_session_entry(user_id, session, id)
        if entry is None:
            return ErrorResponse(message=UNKNOWN_ID_HINT, session_id=session_id)
        # Describing answers to the gate that running does. An MCP
        # description is not a local lookup -- it connects to the server to
        # list its tools -- and a block's schema is what a withheld block
        # was withheld from revealing.
        if entry.kind == "block":
            if gate_denied(BLOCK_GATE):
                return gate_denied_error("blocks", session_id)
            return await _describe_block(entry, user_id, session, expand)
        if entry.kind == "skill":
            if gate_denied(SKILL_TOOL):
                return gate_denied_error(SKILL_TOOL, session_id)
            return describe_skill(entry, session_id)
        if entry.kind == "tool":
            name = entry.implementations[0].ref
            if gate_denied(name):
                return gate_denied_error(name, session_id)
            return _describe_tool(entry, session_id, expand)
        if entry.kind == "mcp_server":
            if gate_denied(MCP_GATE):
                return gate_denied_error("MCP servers", session_id)
            return await _describe_mcp(entry, user_id, session)
        return ErrorResponse(
            message=f"Capabilities of kind '{entry.kind}' cannot be described yet.",
            session_id=session_id,
        )


async def _describe_block(
    entry: CapabilityEntry, user_id: str, session: ChatSession, expand: bool
) -> ToolResponseBase:
    block_id = _block_ref(entry) or ""
    result = await RunBlockTool()._execute(
        user_id, session, block_id=block_id, input_data={}, validate_only=True
    )
    if isinstance(result, BlockDetailsResponse):
        result.message = (
            f"{result.message} Run it with run_capability(id='{entry.id}', "
            "input={...}); credentials are resolved by the platform."
        )
        if not expand:
            result.block.inputs = collapse_large_enums(result.block.inputs)
            result.block.outputs = collapse_large_enums(result.block.outputs)
    return result


def describe_skill(entry: CapabilityEntry, session_id: str) -> ToolResponseBase:
    """What a skill is for.  Its body is not repeated here: running the
    capability loads it, package files included."""
    return CapabilityDetailsResponse(
        message=(
            f"Skill '{entry.name}': {entry.description} Load it with "
            f"run_capability(id='{entry.id}', input={{}}) and read the body "
            "before acting on the task it covers."
        ),
        capability=entry.listing(),
        parameters=NO_INPUT,
        session_id=session_id,
    )


def _describe_tool(
    entry: CapabilityEntry, session_id: str, expand: bool
) -> ToolResponseBase:
    tool = configured_tool(entry.implementations[0].ref)
    if tool is None:
        return ErrorResponse(message=UNKNOWN_ID_HINT, session_id=session_id)
    parameters = tool.parameters if expand else collapse_large_enums(tool.parameters)
    return CapabilityDetailsResponse(
        message=(
            f"{tool.description} Run it with run_capability(id='{entry.id}', "
            "input={...}) where input matches these parameters."
        ),
        capability=entry.listing(),
        parameters=parameters,
        session_id=session_id,
    )


async def _describe_mcp(
    entry: CapabilityEntry, user_id: str, session: ChatSession
) -> ToolResponseBase:
    server_url = entry.implementations[0].ref
    if not server_url:
        return ErrorResponse(
            message=setup_hint(entry.schema_ref or entry.id),
            session_id=session.session_id,
        )
    result = await RunMCPToolTool()._execute(user_id, session, server_url=server_url)
    if isinstance(result, MCPToolsDiscoveredResponse):
        result.message = (
            f"{entry.name} exposes {len(result.tools)} tool(s); `params` lists each "
            "tool's arguments with required ones marked *. Run one with "
            f"run_capability(id='{entry.id}', input={{'tool': <name>, 'arguments': "
            "{...}}). An argument error returns that tool's schema; do not re-list."
        )
    return result


def _block_ref(entry: CapabilityEntry) -> str | None:
    return next(
        (impl.ref for impl in entry.implementations if impl.kind == "block"), None
    )
