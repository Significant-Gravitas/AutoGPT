"""Capability sources: each yields ``CapabilityEntry`` objects from one backing
registry (platform tools, blocks, the MCP catalog).  Per-user sources (MCP
servers the user has registered) are layered on top by the registry."""

from .blocks import block_entries
from .mcp_catalog import mcp_catalog_entries
from .static_tools import EAGER_CORE, RETIRED_TOOLS, tool_entries

__all__ = [
    "EAGER_CORE",
    "RETIRED_TOOLS",
    "block_entries",
    "mcp_catalog_entries",
    "tool_entries",
]
