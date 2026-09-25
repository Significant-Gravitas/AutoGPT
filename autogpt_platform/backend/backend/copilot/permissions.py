"""Copilot execution permissions — tool and block allow/deny filtering.

:class:`CopilotPermissions` is the single model used everywhere:

- ``AutoPilotBlock`` reads four block-input fields and builds one instance.
- ``stream_chat_completion_sdk`` applies it when constructing
  ``ClaudeAgentOptions.allowed_tools`` / ``disallowed_tools``.
- ``run_capability`` reads it from the contextvar to gate block execution.
- Recursive (sub-agent) invocations merge parent and child so children
  can only be *more* restrictive, never more permissive.

Tool names
----------
Users specify the **short name** as it appears in ``TOOL_REGISTRY`` (e.g.
``run_capability``, ``web_fetch``) or as an SDK built-in (e.g. ``Read``,
``Task``, ``WebSearch``).  Internally these are mapped to the full SDK
format (``mcp__copilot__run_capability``, ``Read``, …) by
:func:`apply_tool_permissions`.

Two names are **capability gates** rather than tools: ``run_block`` and
``run_mcp_tool`` no longer exist as tools (blocks and MCP servers run through
``run_capability``), but denying them still denies that whole kind of
capability, so saved graphs keep their meaning.  Retired discovery names
(``find_block``, ``get_mcp_guide``, ``continue_run_block``) are accepted for
saved graphs and mapped to their registry equivalents.

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

Denying a capability denies the tools that extend it (see
``_IMPLIED_DENIALS``); allowing a capability gate allows the tool that now
performs it (see ``_IMPLIED_GRANTS``). Either list is written against the
tools that exist when it is written, so a later tool that reaches the same
resource would otherwise be silently regained by every existing blacklist,
and a renamed one silently lost by every existing whitelist.

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
from typing import TYPE_CHECKING, Literal, get_args

from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from backend.copilot.tools import ToolGroup

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
    "browser_complete_link_payment",
    "browser_link_payment_status",
    "browser_navigate",
    "browser_request_link_payment",
    "browser_reset_after_payment",
    "browser_screenshot",
    "confirm_expert_change",
    "confirm_expert_soul_update",
    "connect_integration",
    "consult_teammate",
    "create_agent",
    "create_feature_request",
    "create_folder",
    "customize_agent",
    "decompose_goal",
    "delegate_to_expert",
    "delete_folder",
    "delete_preset",
    "delete_schedule",
    "delete_skill",
    "delete_workspace_file",
    "describe_capability",
    "edit_agent",
    "edit_chat_platform_message",
    "enter_agent_building_mode",
    "expert_onboarding",
    "find_agent",
    "find_capability",
    "find_library_agent",
    "find_session",
    "fix_agent_graph",
    "get_agent_building_guide",
    "get_doc_page",
    "get_platform_info",
    "get_sub_session_result",
    "grant_expert_credential",
    "handoff_to_expert",
    "hire_expert",
    "install_expert_workflow",
    "list_agent_triggers",
    "list_chat_platform_channels",
    "list_expert_chats",
    "list_expert_credentials",
    "list_expert_workflows",
    "list_folders",
    "list_presets",
    "list_routines",
    "list_schedules",
    "list_skills",
    "list_team",
    "list_workspace_files",
    "memory_forget_confirm",
    "memory_forget_search",
    "memory_search",
    "memory_store",
    "message_session",
    "move_agents_to_folder",
    "move_folder",
    "pause_schedule",
    "post_to_chat_platform",
    "raise_expert",
    "read_expert_chat",
    "read_skill",
    "read_workspace_file",
    "remove_expert_workflow",
    "request_credential_grant",
    "resume_capability",
    "resume_schedule",
    "revoke_expert_credential",
    "run_agent",
    "run_capability",
    "run_sub_session",
    "schedule_followup",
    "schedule_routine",
    "search_docs",
    "search_feature_requests",
    "setup_agent_webhook_trigger",
    "start_desktop",
    "store_skill",
    "update_expert",
    "update_expert_soul",
    "update_folder",
    "update_preset",
    "validate_agent_graph",
    "view_agent_output",
    "web_fetch",
    "web_search",
    "write_workspace_file",
    # Capability gates (not tools): deny to withhold a whole kind of capability
    "run_block",
    "run_mcp_tool",
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

# Capability gates: names a permission list may deny to withhold every block
# (``run_block``) or every MCP server (``run_mcp_tool``) from ``run_capability``.
BLOCK_GATE = "run_block"
MCP_GATE = "run_mcp_tool"
CAPABILITY_GATE_NAMES: frozenset[str] = frozenset({BLOCK_GATE, MCP_GATE})

# Retired tool names -> the registry tool that replaced them.  Accepted in
# saved ``AutoPilotBlock`` permission lists and translated on evaluation.
LEGACY_TOOL_ALIASES: dict[str, str] = {
    "find_block": "find_capability",
    "get_mcp_guide": "find_capability",
    "continue_run_block": "resume_capability",
}

DISABLED_LEGACY_TOOL_NAMES: frozenset[str] = frozenset(LEGACY_TOOL_ALIASES)


# What a routine reaches when nobody has bound it to anything.  A roster
# template is read by whoever reviews the PR, not by the owner whose account it
# will run on, so a seeded routine ships able to research, think, read its own
# workspace and write to its own thread — and nothing else.  Denying the names
# that carry a credential outward is what makes "a cadence may only carry work
# that acts on nothing outside the platform" (``PreloadSeed.cron``) a boundary
# rather than a comment.  An owner who wants their queue swept says so when they
# switch the routine on, and that answer is what lifts this.
#
# The grant is the whole rule at fire time.  Where the prompt came from decides
# what that grant STARTS as — a template arrives ungranted, and a routine the
# owner dictated in their own chat arrives granted, because the same words typed
# into the same chat already run with every tool here and taking ``run_agent``
# off somebody's own morning briefing protects nobody.  Both remain the owner's
# to change, and neither reads as permission on its own.
UNGRANTED_ROUTINE_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        # The two capability gates, so denying them withholds every block and
        # every MCP server rather than one tool's worth of them.
        BLOCK_GATE,
        MCP_GATE,
        "run_agent",
        "post_to_chat_platform",
        # ``bash_exec`` is the one that looks harmless and is not. On E2B the
        # sandbox is handed ``get_integration_env_vars(user_id)`` — the
        # owner's live GH_TOKEN and friends — keyed on the user alone, with no
        # reference to this filter or to the grant. A shell plus internet
        # access plus the owner's tokens in ``env`` is the whole mute undone by
        # one ``echo $GH_TOKEN``, so an ungranted routine does not get a shell.
        "bash_exec",
    }
)


# Refused on EVERY routine turn, however the routine was written and whatever
# its owner granted it.  An unattended turn reads things nobody is watching it
# read — an issue tracker, an inbox, a web page — and text in any of them can
# ask it to schedule more work.  Without this, one injected page buys standing
# access to the account forever: a routine that can write a routine can grant
# itself the credentials its own prompt was denied, and the owner sees a
# schedule they never agreed to.  Standing work is created where somebody is
# present to refuse it.
ROUTINE_SELF_ESCALATION_TOOLS: frozenset[str] = frozenset(
    {
        "schedule_routine",
        "schedule_followup",
        "setup_agent_webhook_trigger",
    }
)


def routine_disabled_tools(*, granted: bool) -> frozenset[str]:
    """Tools to refuse on a routine's unattended turn.

    *granted* is the owner's answer to "may this routine use my connected
    services", which they give per routine when they switch it on.  It only ever
    removes the outward-reaching denials; nothing makes a routine able to
    schedule more of itself.

    Deliberately a denylist of what reaches *outward*, not a narrow allowlist:
    reading, searching, and drafting into the thread are the whole point of an
    unattended routine, and a routine that can do none of those is not worth
    shipping off.

    Kept here rather than beside the other tool gates in ``tools/__init__``:
    the scheduler needs it at import time, and ``executor.scheduler`` importing
    ``copilot.tools`` closes a cycle (tools -> helpers -> executor -> scheduler).
    ``routines_test`` asserts every name here is a live tool or gate.
    """
    if granted:
        return ROUTINE_SELF_ESCALATION_TOOLS
    return UNGRANTED_ROUTINE_DENIED_TOOLS | ROUTINE_SELF_ESCALATION_TOOLS


"""Tool names accepted only for backwards compatibility with saved graphs.

These names are intentionally absent from ``ToolName`` and
``PLATFORM_TOOL_NAMES`` so they are not exposed in new block schemas or sent to
the model as available tools.
"""

# SDK built-in tool names — tools provided by the Claude Code CLI that our
# code does not implement directly.  ``TodoWrite`` is DELIBERATELY excluded:
# baseline mode ships an MCP-wrapped platform version
# (``tools/todo_write.py``), while SDK mode still uses the CLI-native
# original via ``_SDK_BUILTIN_ALWAYS`` in ``sdk/tool_adapter.py`` — the
# MCP copy is never registered there (``BASELINE_ONLY_MCP_TOOLS``).
# ``Task`` remains an SDK-only built-in
# (for queue-backed context-isolation on baseline, use ``run_sub_session``
# instead).
SDK_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {"Agent", "Edit", "Glob", "Grep", "Read", "Task", "WebSearch", "Write"}
)

# Platform tool names — everything that isn't an SDK built-in or a gate.
PLATFORM_TOOL_NAMES: frozenset[str] = (
    ALL_TOOL_NAMES - SDK_BUILTIN_TOOL_NAMES - CAPABILITY_GATE_NAMES
)

# Compiled regex patterns for block identifier classification.
_FULL_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_PARTIAL_UUID_RE = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


# Tools that a blacklist entry must deny alongside the capability named.
#
# A blacklist is written against the tools that existed when it was written,
# so a tool added later that reaches the same resource is silently regained
# by every blacklist already out there. An operator who revoked proactive
# posting to their chat platforms should not find the agent able to rewrite
# everything the bot has already said in them.
_IMPLIED_DENIALS: dict[str, tuple[str, ...]] = {
    "post_to_chat_platform": ("edit_chat_platform_message",),
}


def _with_implied_denials(denied: frozenset[str]) -> frozenset[str]:
    """Expand a deny set with the tools its entries imply."""
    return denied.union(
        implied for name in denied for implied in _IMPLIED_DENIALS.get(name, ())
    )


# Tools that a whitelist entry must allow alongside the capability named.
#
# The mirror of the above: a whitelist is also written against the tools that
# existed when it was written, and the gates are the one case where the tool
# that does the work was renamed out from under it. A saved list naming
# ``run_block`` meant "you may run blocks", which now happens through
# ``run_capability`` -- a name no existing list can contain. Without this the
# gate stays open and the tool that opens it is never handed to the model.
_IMPLIED_GRANTS: dict[str, tuple[str, ...]] = {
    BLOCK_GATE: ("run_capability",),
    MCP_GATE: ("run_capability",),
}


def _with_implied_grants(allowed: frozenset[str]) -> frozenset[str]:
    """Expand an allow set with the tools its entries imply.

    A deferred tool is no longer handed to the model, so allowing it without
    ``run_capability`` allows nothing: the dispatcher is the only way in.
    """
    grants = allowed.union(
        implied for name in allowed for implied in _IMPLIED_GRANTS.get(name, ())
    )
    if grants & _deferred_tool_names():
        grants = grants | {"run_capability"}
    return grants


def _deferred_tool_names() -> frozenset[str]:
    """The registry's deferred set, imported late (heavy tool imports)."""
    from backend.copilot.tools import DEFERRED_TOOL_NAMES  # noqa: PLC0415

    return DEFERRED_TOOL_NAMES


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
        tools: Tool names to filter (short names, e.g. ``run_capability``).
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
        tool_set = frozenset(LEGACY_TOOL_ALIASES.get(t, t) for t in self.tools)
        if self.tools_exclude:
            return all_tools - _with_implied_denials(tool_set)
        return all_tools & _with_implied_grants(tool_set)

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
    return [
        t
        for t in tools
        if t not in ALL_TOOL_NAMES and t not in DISABLED_LEGACY_TOOL_NAMES
    ]


_tool_names_checked = False


def denied_tool_names(permissions: CopilotPermissions | None) -> frozenset[str]:
    """Short names *permissions* withholds from the turn (empty when none)."""
    if permissions is None or permissions.is_empty():
        return frozenset()
    all_tools = all_known_tool_names()
    return all_tools - permissions.effective_allowed_tools(all_tools)


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
    disabled_groups: Iterable[ToolGroup] = (),
) -> tuple[list[str], list[str]]:
    """Compute (allowed_tools, extra_disallowed) for :class:`ClaudeAgentOptions`.

    Takes the base allowed/disallowed lists from
    :func:`~backend.copilot.sdk.tool_adapter.get_copilot_tool_names` /
    :func:`~backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools` and
    applies *permissions* on top.  Tools belonging to any *disabled_groups*
    are hidden from the base allowed list — use this to gate capability
    groups (e.g. ``"graphiti"`` when the memory backend is off for the
    current user).

    Returns:
        ``(allowed_tools, extra_disallowed)`` where *allowed_tools* is the
        possibly-narrowed list to pass to ``ClaudeAgentOptions.allowed_tools``
        and *extra_disallowed* is the list to pass to
        ``ClaudeAgentOptions.disallowed_tools``.
    """
    from backend.copilot.sdk.tool_adapter import (
        _READ_TOOL_NAME,
        BASELINE_ONLY_MCP_TOOLS,
        MCP_TOOL_PREFIX,
        get_copilot_tool_names,
        get_sdk_disallowed_tools,
    )
    from backend.copilot.tools import TOOL_REGISTRY

    base_allowed = get_copilot_tool_names(
        use_e2b=use_e2b, disabled_groups=disabled_groups
    )
    base_disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)

    if permissions.is_empty():
        return base_allowed, base_disallowed

    all_tools = all_known_tool_names()
    effective = permissions.effective_allowed_tools(all_tools)

    # SDK built-in file tools are replaced by MCP equivalents in both modes.
    # Map each SDK built-in name to its MCP tool name so users can use the
    # familiar names in their permissions and the correct tools are included.
    _SDK_TO_MCP: dict[str, str] = {}
    if use_e2b:
        from backend.copilot.sdk.e2b_file_tools import E2B_FILE_TOOL_NAMES

        _SDK_TO_MCP = dict(
            zip(
                ["Read", "Write", "Edit", "Glob", "Grep"],
                E2B_FILE_TOOL_NAMES,
                strict=False,
            )
        )
    else:
        from backend.copilot.sdk.e2b_file_tools import EDIT_TOOL_NAME as _EDIT
        from backend.copilot.sdk.e2b_file_tools import READ_TOOL_NAME as _READ
        from backend.copilot.sdk.e2b_file_tools import WRITE_TOOL_NAME as _WRITE

        _SDK_TO_MCP = {"Read": _READ, "Write": _WRITE, "Edit": _EDIT}

    # Build an updated allowed list by mapping short names → SDK names and
    # keeping only those present in the original base_allowed list.
    def to_sdk_names(short: str) -> list[str]:
        names: list[str] = []
        if short in BASELINE_ONLY_MCP_TOOLS:
            # Baseline ships MCP versions of these (Task/TodoWrite) for
            # model-flexibility parity, but SDK mode uses the CLI-native
            # originals. Permissions target the CLI built-in here so
            # ``base_allowed`` (which excludes the MCP wrappers) still
            # matches.
            names.append(short)
        elif short in TOOL_REGISTRY:
            names.append(f"{MCP_TOOL_PREFIX}{short}")
        elif short in _SDK_TO_MCP:
            # Offer BOTH spellings and let the ``base_allowed`` filter below
            # pick the one this mode registers: outside E2B only ``read_file``
            # has an MCP wrapper, and the MCP spelling of Write/Edit is not in
            # ``base_allowed``, so mapping them solely to it drops them from
            # every filtered turn.
            names.append(f"{MCP_TOOL_PREFIX}{_SDK_TO_MCP[short]}")
            names.append(short)
        else:
            names.append(short)  # SDK built-in — used as-is
        return names

    # short names permitted by permissions
    permitted_sdk: set[str] = set()
    for s in effective:
        permitted_sdk.update(to_sdk_names(s))
    # Always include the internal read_tool_result tool (used by SDK for large/truncated outputs)
    permitted_sdk.add(f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}")

    filtered_allowed = [t for t in base_allowed if t in permitted_sdk]

    # Extra disallowed = tools that were in base_allowed but are now removed
    removed = set(base_allowed) - set(filtered_allowed)
    extra_disallowed = list(set(base_disallowed) | removed)

    return filtered_allowed, extra_disallowed


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# Dream-pass sub-agent capability filter (spec: ``dream/p0-spec.md`` §P0.5).
#
# The dream orchestrator runs phases 1/2/3 via direct LLM calls today, but P0.5
# (web fact-check) and P9 (daydreaming) hand work to a sub-agent that must be
# strictly capability-bounded: read/write the user's Graphiti memory and
# fact-check claims against the web — nothing else.  No shell, no general web
# browsing, no file ops, no agent-graph mutation.
#
# Stored as a **whitelist** (``tools_exclude=False``) so the universe of
# permitted tools is explicit and grows only when this preset is edited.
#
# ``web_fact_check`` is intentionally listed even though the tool does not yet
# exist in ``TOOL_REGISTRY`` — it lands in P0.5.  Unknown short names are
# silently dropped by :meth:`CopilotPermissions.effective_allowed_tools` (it
# intersects with ``all_tools``), so the preset is forward-compatible without
# needing to be updated when the tool ships.
DREAM_PERMISSIONS: CopilotPermissions = CopilotPermissions(
    tools=[
        "memory_search",
        "memory_store",
        "memory_forget_search",
        "memory_forget_confirm",
        "web_fact_check",
    ],
    tools_exclude=False,
)
