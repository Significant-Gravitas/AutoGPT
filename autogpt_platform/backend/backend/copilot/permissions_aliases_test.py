"""Legacy tool names in saved permission lists keep their meaning: retired
discovery names alias to the registry tools, and ``run_block`` /
``run_mcp_tool`` survive as capability gates."""

from backend.copilot.permissions import (
    ALL_TOOL_NAMES,
    BLOCK_GATE,
    CAPABILITY_GATE_NAMES,
    MCP_GATE,
    PLATFORM_TOOL_NAMES,
    CopilotPermissions,
    denied_tool_names,
    validate_tool_names,
)
from backend.copilot.tools import TOOL_REGISTRY


def test_gates_are_permission_names_but_not_tools():
    assert CAPABILITY_GATE_NAMES <= ALL_TOOL_NAMES
    assert CAPABILITY_GATE_NAMES.isdisjoint(PLATFORM_TOOL_NAMES)
    assert BLOCK_GATE not in TOOL_REGISTRY and MCP_GATE not in TOOL_REGISTRY
    assert PLATFORM_TOOL_NAMES == frozenset(TOOL_REGISTRY)


def test_legacy_whitelist_maps_to_registry_tools():
    perms = CopilotPermissions(
        tools=["find_block", "run_block", "continue_run_block"], tools_exclude=False
    )
    allowed = perms.effective_allowed_tools(ALL_TOOL_NAMES)
    assert {"find_capability", "resume_capability", BLOCK_GATE} <= allowed
    assert "run_capability" not in allowed  # blocks run through the gate only


def test_legacy_blacklist_denies_the_gate():
    perms = CopilotPermissions(tools=["run_block"])
    allowed = perms.effective_allowed_tools(ALL_TOOL_NAMES)
    assert BLOCK_GATE not in allowed and "run_capability" in allowed
    assert denied_tool_names(perms) == frozenset({BLOCK_GATE})


def test_legacy_names_validate():
    assert (
        validate_tool_names(["find_block", "get_mcp_guide", "continue_run_block"]) == []
    )
    assert validate_tool_names(["not_a_tool"]) == ["not_a_tool"]


def test_denied_tool_names_empty_without_filter():
    assert denied_tool_names(None) == frozenset()
    assert denied_tool_names(CopilotPermissions()) == frozenset()
