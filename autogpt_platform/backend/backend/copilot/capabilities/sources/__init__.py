"""Capability sources: each yields ``CapabilityEntry`` objects from one backing
registry (platform tools, blocks, the MCP catalog, the session owner's
skills, the expert roster and team).  The first three make the process-wide
registry; skills and experts are per session and layered on top by
``tools.session_registry``."""

from .blocks import block_entries
from .experts import (
    DELEGATE_TOOL,
    EXPERT_ID_PREFIX,
    HIRE_TOOL,
    TEAMMATE_ID_PREFIX,
    expert_dispatch,
    expert_entries,
)
from .mcp_catalog import mcp_catalog_entries
from .skills import SKILL_ID_PREFIX, SKILL_TOOL, skill_entries, skill_name
from .static_tools import EAGER_CORE, RETIRED_TOOLS, tool_entries

__all__ = [
    "DELEGATE_TOOL",
    "EAGER_CORE",
    "EXPERT_ID_PREFIX",
    "HIRE_TOOL",
    "TEAMMATE_ID_PREFIX",
    "RETIRED_TOOLS",
    "SKILL_ID_PREFIX",
    "SKILL_TOOL",
    "block_entries",
    "expert_dispatch",
    "expert_entries",
    "mcp_catalog_entries",
    "skill_entries",
    "skill_name",
    "tool_entries",
]
