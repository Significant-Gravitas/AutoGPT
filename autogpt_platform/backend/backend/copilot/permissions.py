"""Copilot execution permissions — tool and block allow/deny filtering.

:class:`CopilotPermissions` is the single model used everywhere:

- ``AutoPilotBlock`` reads four block-input fields and builds one instance.
- ``stream_chat_completion_sdk`` applies it when constructing
  ``ClaudeAgentOptions.allowed_tools`` / ``disallowed_tools``.
- ``run_block`` reads it from the contextvar to gate block execution.
- Recursive (sub-agent) invocations merge parent and child so children
  can only be *more* restrictive, never more permissive.

Tool names
----------
Users specify the **short name** as it appears in ``TOOL_REGISTRY`` (e.g.
``run_block``, ``web_fetch``) or as an SDK built-in (e.g. ``Read``,
``Task``, ``WebSearch``).  Internally these are mapped to the full SDK
format (``mcp__copilot__run_block``, ``Read``, …) by
:func:`apply_tool_permissions`.

Block identifiers
-----------------
Each entry in ``blocks`` may be one of:

- A **full UUID** (``c069dc6b-c3ed-4c12-b6e5-d47361e64ce6``)
- A **partial UUID** — the first 8-character hex segment (``c069dc6b``)
- A **block name** (case-insensitive, e.g. ``"HTTP Request"``)

:func:`validate_block_identifiers` resolves all entries against the live
block registry and returns any that could not be matched.

Semantics
---------
``tools_exclude=True``  (default) — ``tools`` is a **blacklist**; listed
tools are denied and everything else is allowed.  An empty list means
"allow all" (no filtering).

``tools_exclude=False`` — ``tools`` is a **whitelist**; only listed tools
are allowed.

``blocks_exclude`` follows the same pattern for ``blocks``.

Recursion inheritance
---------------------
:meth:`CopilotPermissions.merged_with_parent` produces a new instance that
is at most as permissive as the parent:

- Tools: effective-allowed sets are intersected then stored as a whitelist.
- Blocks: the parent is stored in ``_parent`` and consulted during every
  :meth:`is_block_allowed` call so both constraints must pass.
"""

from __future__ import annotations

import re
from typing import Literal, get_args

from pydantic import BaseModel, PrivateAttr

# ---------------------------------------------------------------------------
# Constants — single source of truth for all accepted tool names
# ---------------------------------------------------------------------------

# Literal type combining all valid tool names — used by AutoPilotBlock.Input
# so the frontend renders a multi-select dropdown.
# This is the SINGLE SOURCE OF TRUTH.  All other name sets are derived from it.
ToolName = Literal[
    # Platform tools (must match keys in TOOL_REGISTRY)
    "add_understanding",
    "ask_question",
    "bash_exec",
    "browser_act",
    "browser_navigate",
    "browser_screenshot",
    "connect_integration",
    "continue_run_block",
    "create_agent",
    "create_feature_request",
    "create_folder",
    "customize_agent",
    "delete_folder",
    "delete_workspace_file",
    "edit_agent",
    "find_agent",
    "find_block",
    "find_library_agent",
    "fix_agent_graph",
    "get_agent_building_guide",
    "get_doc_page",
    "get_mcp_guide",
    "list_folders",
    "list_workspace_files",
    "memory_search",
    "memory_store",
    "move_agents_to_folder",
    "move_folder",
    "read_workspace_file",
    "run_agent",
    "run_block",
    "run_mcp_tool",
    "search_docs",
    "search_feature_requests",
    "update_folder",
    "validate_agent_graph",
    "view_agent_output",
    "web_fetch",
    "write_workspace_file",
    # SDK built-ins
    "Agent",
    "Edit",
    "Glob",
    "Grep",
    "Read",
    "Task",
    "TodoWrite",
    "WebSearch",
    "Write",
]

# Frozen set of all valid tool names — derived from the Literal.
ALL_TOOL_NAMES: frozenset[str] = frozenset(get_args(ToolName))

# SDK built-in tool names — uppercase-initial names are SDK built-ins.
SDK_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    n for n in ALL_TOOL_NAMES if n[0].isupper()
)

# Platform tool names — everything that isn't an SDK built-in.
PLATFORM_TOOL_NAMES: frozenset[str] = ALL_TOOL_NAMES - SDK_BUILTIN_TOOL_NAMES

# Compiled regex patterns for block identifier classification.
_FULL_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_PARTIAL_UUID_RE = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper — block identifier matching
# ---------------------------------------------------------------------------


def _block_matches(identifier: str, block_id: str, block_name: str) -> bool:
    """Return True if *identifier* resolves to the given block.

    Resolution order:
    1. Full UUID — exact case-insensitive match against *block_id*.
    2. Partial UUID (8 hex chars, first segment) — prefix match.
    3. Name — case-insensitive equality against *block_name*.
    """
    ident = identifier.strip()
    if _FULL_UUID_RE.match(ident):
        return ident.lower() == block_id.lower()
    if _PARTIAL_UUID_RE.match(ident):
        return block_id.lower().startswith(ident.lower())
    return ident.lower() == block_name.lower()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CopilotPermissions(BaseModel):
    """Capability filter for a single copilot execution.

    Attributes:
        tools: Tool names to filter (short names, e.g. ``run_block``).
        tools_exclude: When True (default) ``tools`` is a blacklist;
            when False it is a whitelist.  Ignored when *tools* is empty.
        blocks: Block identifiers (name, full UUID, or 8-char partial UUID).
        blocks_exclude: Same semantics as *tools_exclude* but for blocks.
    """

    tools: list[str] = []
    tools_exclude: bool = True
    blocks: list[str] = []
    blocks_exclude: bool = True

    # Private: parent permissions for recursion inheritance.
    # Set only by merged_with_parent(); never exposed in block input schema.
    _parent: CopilotPermissions | None = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def effective_allowed_tools(self, all_tools: frozenset[str]) -> frozenset[str]:
        """Compute the set of short tool names that are permitted.

        Args:
            all_tools: Universe of valid short tool names.

        Returns:
            Subset of *all_tools* that pass the filter.
        """
        if not self.tools:
            return frozenset(all_tools)
        tool_set = frozenset(self.tools)
        if self.tools_exclude:
            return all_tools - tool_set
        return all_tools & tool_set

    # ------------------------------------------------------------------
    # Block helpers
    # ------------------------------------------------------------------

    def is_block_allowed(self, block_id: str, block_name: str) -> bool:
        """Return True if the block may be executed under these permissions.

        Checks this instance first, then consults the parent (if any) so
        the entire inheritance chain is respected.
        """
        if not self._check_block_locally(block_id, block_name):
            return False
        if self._parent is not None:
            return self._parent.is_block_allowed(block_id, block_name)
        return True

    def _check_block_locally(self, block_id: str, block_name: str) -> bool:
        """Check *only* this instance's block filter (ignores parent)."""
        if not self.blocks:
            return True  # No filter → allow all
        matched = any(
            _block_matches(identifier, block_id, block_name)
            for identifier in self.blocks
        )
        return not matched if self.blocks_exclude else matched

    # ------------------------------------------------------------------
    # Recursion / merging
    # ------------------------------------------------------------------

    def merged_with_parent(
        self,
        parent: CopilotPermissions,
        all_tools: frozenset[str],
    ) -> CopilotPermissions:
        """Return a new instance that is at most as permissive as *parent*.

        - Tools: intersection of effective-allowed sets, stored as a whitelist.
        - Blocks: parent is stored internally; both constraints are applied
          during :meth:`is_block_allowed`.
        """
        merged_tools = self.effective_allowed_tools(
            all_tools
        ) & parent.effective_allowed_tools(all_tools)
        result = CopilotPermissions(
            tools=sorted(merged_tools),
            tools_exclude=False,
            blocks=self.blocks,
            blocks_exclude=self.blocks_exclude,
        )
        result._parent = parent
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True when no filtering is configured (allow-all passthrough)."""
        return not self.tools and not self.blocks and self._parent is None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def all_known_tool_names() -> frozenset[str]:
    """Return all short tool names accepted in *tools*.

    Returns the pre-computed ``ALL_TOOL_NAMES`` set (derived from the
    ``ToolName`` Literal).  On first call, also verifies consistency with
    the live ``TOOL_REGISTRY``.
    """
    _assert_tool_names_consistent()
    return ALL_TOOL_NAMES


def validate_tool_names(tools: list[str]) -> list[str]:
    """Return entries in *tools* that are not valid tool names.

    Args:
        tools: List of short tool name strings to validate.

    Returns:
        List of invalid names (empty if all are valid).
    """
    return [t for t in tools if t not in ALL_TOOL_NAMES]


_tool_names_checked = False


def _assert_tool_names_consistent() -> None:
    """Verify that ``PLATFORM_TOOL_NAMES`` matches ``TOOL_REGISTRY`` keys.

    Called once lazily (TOOL_REGISTRY has heavy imports).  Raises
    ``AssertionError`` with a helpful diff if they diverge.
    """
    global _tool_names_checked
    if _tool_names_checked:
        return
    _tool_names_checked = True

    from backend.copilot.tools import TOOL_REGISTRY

    registry_keys: frozenset[str] = frozenset(TOOL_REGISTRY.keys())
    declared: frozenset[str] = PLATFORM_TOOL_NAMES
    if registry_keys != declared:
        missing = registry_keys - declared
        extra = declared - registry_keys
        parts: list[str] = [
            "PLATFORM_TOOL_NAMES in permissions.py is out of sync with TOOL_REGISTRY."
        ]
        if missing:
            parts.append(f"  Missing from PLATFORM_TOOL_NAMES: {sorted(missing)}")
        if extra:
            parts.append(f"  Extra in PLATFORM_TOOL_NAMES: {sorted(extra)}")
        parts.append("  Update the ToolName Literal to match.")
        raise AssertionError("\n".join(parts))


async def validate_block_identifiers(
    identifiers: list[str],
) -> list[str]:
    """Resolve each block identifier and return those that could not be matched.

    Args:
        identifiers: List of block identifiers (name, full UUID, or partial UUID).

    Returns:
        List of identifiers that matched no known block.
    """
    from backend.blocks import get_blocks

    # get_blocks() returns dict[block_id_str, BlockClass]; instantiate once to get names.
    block_registry = get_blocks()
    block_info = {bid: cls().name for bid, cls in block_registry.items()}
    invalid: list[str] = []
    for ident in identifiers:
        matched = any(
            _block_matches(ident, bid, bname) for bid, bname in block_info.items()
        )
        if not matched:
            invalid.append(ident)
    return invalid


# ---------------------------------------------------------------------------
# SDK tool-list application
# ---------------------------------------------------------------------------


def apply_tool_permissions(
    permissions: CopilotPermissions,
    *,
    use_e2b: bool = False,
) -> tuple[list[str], list[str]]:
    """Compute (allowed_tools, extra_disallowed) for :class:`ClaudeAgentOptions`.

    Takes the base allowed/disallowed lists from
    :func:`~backend.copilot.sdk.tool_adapter.get_copilot_tool_names` /
    :func:`~backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools` and
    applies *permissions* on top.

    Returns:
        ``(allowed_tools, extra_disallowed)`` where *allowed_tools* is the
        possibly-narrowed list to pass to ``ClaudeAgentOptions.allowed_tools``
        and *extra_disallowed* is the list to pass to
        ``ClaudeAgentOptions.disallowed_tools``.
    """
    from backend.copilot.sdk.tool_adapter import (
        _READ_TOOL_NAME,
        MCP_TOOL_PREFIX,
        get_copilot_tool_names,
        get_sdk_disallowed_tools,
    )
    from backend.copilot.tools import TOOL_REGISTRY

    base_allowed = get_copilot_tool_names(use_e2b=use_e2b)
    base_disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)

    if permissions.is_empty():
        return base_allowed, base_disallowed

    all_tools = all_known_tool_names()
    effective = permissions.effective_allowed_tools(all_tools)

    # In E2B mode, SDK built-in file tools (Read, Write, Edit, Glob, Grep)
    # are replaced by MCP equivalents (read_file, write_file, ...).
    # Map each SDK built-in name to its E2B MCP name so users can use the
    # familiar names in their permissions and the E2B tools are included.
    _SDK_TO_E2B: dict[str, str] = {}
    if use_e2b:
        from backend.copilot.sdk.e2b_file_tools import E2B_FILE_TOOL_NAMES

        _SDK_TO_E2B = dict(
            zip(
                ["Read", "Write", "Edit", "Glob", "Grep"],
                E2B_FILE_TOOL_NAMES,
                strict=False,
            )
        )

    # Build an updated allowed list by mapping short names → SDK names and
    # keeping only those present in the original base_allowed list.
    def to_sdk_names(short: str) -> list[str]:
        names: list[str] = []
        if short in TOOL_REGISTRY:
            names.append(f"{MCP_TOOL_PREFIX}{short}")
        elif short in _SDK_TO_E2B:
            # E2B mode: map SDK built-in file tool to its MCP equivalent.
            names.append(f"{MCP_TOOL_PREFIX}{_SDK_TO_E2B[short]}")
        else:
            names.append(short)  # SDK built-in — used as-is
        return names

    # short names permitted by permissions
    permitted_sdk: set[str] = set()
    for s in effective:
        permitted_sdk.update(to_sdk_names(s))
    # Always include the internal Read tool (used by SDK for large/truncated outputs)
    permitted_sdk.add(f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}")

    filtered_allowed = [t for t in base_allowed if t in permitted_sdk]

    # Extra disallowed = tools that were in base_allowed but are now removed
    removed = set(base_allowed) - set(filtered_allowed)
    extra_disallowed = list(set(base_disallowed) | removed)

    return filtered_allowed, extra_disallowed
