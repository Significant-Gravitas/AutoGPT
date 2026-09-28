"""Capability sources: each yields ``CapabilityEntry`` objects from one backing
registry (platform tools, blocks, the MCP catalog, the session owner's
skills).  The first three make the process-wide registry; skills are per
session and layered on top by ``tools.session_registry``."""

from .blocks import block_entries
from .mcp_catalog import mcp_catalog_entries
from .skills import SKILL_ID_PREFIX, SKILL_TOOL, skill_entries, skill_name
from .static_tools import EAGER_CORE, RETIRED_TOOLS, tool_entries

__all__ = [
    "EAGER_CORE",
    "RETIRED_TOOLS",
    "SKILL_ID_PREFIX",
    "SKILL_TOOL",
    "block_entries",
    "mcp_catalog_entries",
    "skill_entries",
    "skill_name",
    "tool_entries",
]
