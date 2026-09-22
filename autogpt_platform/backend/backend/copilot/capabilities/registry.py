"""Assemble the capability index from its sources.

The platform-wide index (tools + blocks + MCP catalog) is static for the
life of the process, so it is built once and reused.  Per-session entries
(the owner's skills) are layered on top per call by
``tools.session_registry``; nothing here depends on the user.

``backend.copilot.tools`` imports the registry tools, which import this
package, so the tools package hands over its registry with
:func:`configure_tools` once it has built it instead of this module
importing it.
"""

import logging
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.sources import (
    block_entries,
    mcp_catalog_entries,
    tool_entries,
)

if TYPE_CHECKING:
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
_tools: "Mapping[str, BaseTool] | None" = None
_groups: Mapping[str, str] = {}


def configure_tools(tools: "Mapping[str, BaseTool]", groups: Mapping[str, str]) -> None:
    """Hand the platform tool registry to this package (called once by
    ``backend.copilot.tools`` after ``TOOL_REGISTRY`` exists)."""
    global _tools, _groups
    with _lock:
        _tools, _groups = tools, groups


def configured_tool(name: str) -> "BaseTool | None":
    """The platform tool behind a ``tool:`` entry, or None."""
    return None if _tools is None else _tools.get(name)


def get_registry() -> CapabilityIndex:
    """The process-wide platform index, built on first use."""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                if _tools is None:
                    raise RuntimeError(
                        "Capability registry used before configure_tools()"
                    )
                _registry = CapabilityIndex(build_entries(_tools, _groups))
                logger.info(f"Capability registry built: {len(_registry)} entries")
    return _registry


def reset_registry() -> None:
    """Drop the cached index (tests, block reloads)."""
    global _registry
    with _lock:
        _registry = None


def build_entries(
    tools: "Mapping[str, BaseTool]",
    groups: Mapping[str, str],
    *,
    include_blocks: bool = True,
    include_catalog: bool = True,
    include_disabled_blocks: bool = False,
) -> list[CapabilityEntry]:
    entries = tool_entries(tools, groups)
    if include_blocks:
        entries += block_entries(include_disabled=include_disabled_blocks)
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
        tool.description = " ".join(
            text for text in (tool.description, block.description) if text
        )
        tool.tags = sorted(set(tool.tags) | set(block.tags))
        tool.argument_names = list(
            dict.fromkeys(tool.argument_names + block.argument_names)
        )
        folded.add(block.id)
    return [entry for entry in by_id.values() if entry.id not in folded]
