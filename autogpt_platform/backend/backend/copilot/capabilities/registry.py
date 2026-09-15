"""Assemble the capability index from its sources.

The platform-wide index (tools + blocks + MCP catalog) is static for the
life of the process, so it is built once and reused.  Per-user entries (MCP
servers the user has registered) are layered on top per request by a later
change; nothing here depends on the user.
"""

import logging
import threading
from collections.abc import Mapping

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.sources import (
    block_entries,
    mcp_catalog_entries,
    tool_entries,
)
from backend.copilot.tools.base import BaseTool

logger = logging.getLogger(__name__)

# A tool and a block that are the same capability: the tool serves direct
# use, the block serves agent graphs.  Keyed by tool entry id, valued by the
# block class name folded into it.
MERGED_IMPLEMENTATIONS: dict[str, str] = {
    "tool:bash_exec": "ExecuteCodeBlock",
}

_lock = threading.Lock()
_registry: CapabilityIndex | None = None


def get_registry(
    tools: Mapping[str, BaseTool], groups: Mapping[str, str]
) -> CapabilityIndex:
    """The process-wide platform index, built on first use.

    *tools*/*groups* are ``TOOL_REGISTRY``/``TOOL_GROUPS``; the caller passes
    them because the tools package imports this one.  They only matter on
    the first call.
    """
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = CapabilityIndex(build_entries(tools, groups))
                logger.info(f"Capability registry built: {len(_registry)} entries")
    return _registry


def reset_registry() -> None:
    """Drop the cached index (tests, block reloads)."""
    global _registry
    with _lock:
        _registry = None


def build_entries(
    tools: Mapping[str, BaseTool],
    groups: Mapping[str, str],
    *,
    include_blocks: bool = True,
    include_catalog: bool = True,
) -> list[CapabilityEntry]:
    entries = tool_entries(tools, groups)
    if include_blocks:
        entries += block_entries()
    if include_catalog:
        entries += mcp_catalog_entries()
    return _merge(entries)


def _merge(entries: list[CapabilityEntry]) -> list[CapabilityEntry]:
    by_id = {entry.id: entry for entry in entries}
    if len(by_id) != len(entries):
        logger.warning(f"Duplicate capability ids dropped: {len(entries) - len(by_id)}")
    by_name = {entry.name: entry for entry in entries if entry.kind == "block"}
    folded: set[str] = set()
    for tool_id, block_name in MERGED_IMPLEMENTATIONS.items():
        tool, block = by_id.get(tool_id), by_name.get(block_name)
        if tool is None or block is None:
            continue
        tool.implementations += block.implementations
        tool.context = "both"
        tool.tags = sorted(set(tool.tags) | set(block.tags))
        tool.argument_names = list(
            dict.fromkeys(tool.argument_names + block.argument_names)
        )
        folded.add(block.id)
    return [entry for entry in by_id.values() if entry.id not in folded]
