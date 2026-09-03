"""Keep the MCP tool surface deliberate: every Copilot tool is classified."""

from backend.api.external.v2.mcp_server import EXTERNAL_USE_EXCLUSIONS
from backend.copilot.tools import TOOL_REGISTRY


def test_every_tool_is_either_exposed_or_explicitly_excluded():
    exposed = {name for name, t in TOOL_REGISTRY.items() if t.allow_external_use[0]}
    unclassified = set(TOOL_REGISTRY) - exposed - set(EXTERNAL_USE_EXCLUSIONS)
    assert not unclassified, (
        "Tools neither opted in via allow_external_use nor listed in "
        f"EXTERNAL_USE_EXCLUSIONS: {sorted(unclassified)}"
    )


def test_exclusion_list_has_no_stale_or_contradictory_entries():
    unknown = set(EXTERNAL_USE_EXCLUSIONS) - set(TOOL_REGISTRY)
    assert not unknown, f"Excluded tools that no longer exist: {sorted(unknown)}"

    contradictory = [
        name
        for name in EXTERNAL_USE_EXCLUSIONS
        if TOOL_REGISTRY[name].allow_external_use[0]
    ]
    assert (
        not contradictory
    ), f"Tools both opted in and excluded (drop one): {sorted(contradictory)}"


def test_exposed_tools_declare_permissions_as_a_sequence():
    for name, tool in TOOL_REGISTRY.items():
        allowed, perms = tool.allow_external_use
        if allowed:
            assert perms is not None, f"{name} opted in without a permission list"
