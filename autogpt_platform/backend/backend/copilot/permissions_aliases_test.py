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


def test_allowing_a_gate_allows_the_tool_that_runs_it():
    """A saved whitelist says what the agent may do, not what we named it.

    ``run_block`` meant "you may run blocks"; the tool that runs them is now
    ``run_capability``, which no list written before the rename can contain.
    Keep the gate open but withhold the tool and the agent can search and is
    permitted to run, with nothing to run it with.
    """
    for gate in (BLOCK_GATE, MCP_GATE):
        allowed = CopilotPermissions(
            tools=[gate], tools_exclude=False
        ).effective_allowed_tools(ALL_TOOL_NAMES)
        assert {gate, "run_capability"} <= allowed
    # The grant is implied by the gate, not handed out to every whitelist.
    assert "run_capability" not in CopilotPermissions(
        tools=["find_block"], tools_exclude=False
    ).effective_allowed_tools(ALL_TOOL_NAMES)


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


def test_allowing_a_deferred_tool_grants_the_dispatcher():
    """A deferred tool is not in the model's tool list (#14569), so a list
    naming one and not ``run_capability`` hands the model no way to reach it."""
    from backend.copilot.tools import DEFERRED_TOOL_NAMES

    for name in ("list_schedules", "hire_expert", "memory_store"):
        assert name in DEFERRED_TOOL_NAMES
        allowed = CopilotPermissions(
            tools=[name], tools_exclude=False
        ).effective_allowed_tools(ALL_TOOL_NAMES)
        assert {name, "run_capability"} <= allowed


def test_allowing_an_eager_tool_does_not_grant_the_dispatcher():
    allowed = CopilotPermissions(
        tools=["web_fetch"], tools_exclude=False
    ).effective_allowed_tools(ALL_TOOL_NAMES)
    assert allowed == frozenset({"web_fetch"})
