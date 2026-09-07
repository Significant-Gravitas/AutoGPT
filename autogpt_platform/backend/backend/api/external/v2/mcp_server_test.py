"""Keep the MCP tool surface deliberate: every Copilot tool is classified."""

from mcp.server.fastmcp import FastMCP

from backend.api.external.v2.mcp_server import (
    EXTERNAL_USE_EXCLUSIONS,
    META_KEY_REQUIRED_SCOPES,
    UNSCOPED_EXTERNAL_TOOLS,
    create_mcp_server,
)
from backend.copilot.tools import TOOL_REGISTRY


async def test_the_server_carries_every_opted_in_tool_with_its_scopes():
    """Registration goes through FastMCP's own constructor, not its internals.

    Writing into `_tool_manager._tools` worked by accident of mcp 1.26.0's
    layout; this fails the moment the supported path stops carrying the tools.
    """
    server = create_mcp_server()

    exposed = {name for name, t in TOOL_REGISTRY.items() if t.allow_external_use[0]}
    # The base listing: the subclass's own filters by the caller's scopes.
    registered = {t.name: t for t in await FastMCP.list_tools(server)}
    assert set(registered) == exposed

    for name, tool in registered.items():
        expected = [p.value for p in (TOOL_REGISTRY[name].allow_external_use[1] or [])]
        assert (tool.meta or {}).get(META_KEY_REQUIRED_SCOPES) == expected


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


def test_no_tool_is_exposed_unscoped_without_a_stated_reason():
    """An empty permission list means any key can drive the tool.

    That is right for published docs and public listings and wrong for anything
    that spends platform money or acts through a platform-owned account, so the
    open set is enumerated rather than inferred.
    """
    unscoped = {
        name
        for name, tool in TOOL_REGISTRY.items()
        if tool.allow_external_use[0] and not tool.allow_external_use[1]
    }
    assert unscoped == set(UNSCOPED_EXTERNAL_TOOLS), (
        "unlisted tools exposed with no permission: "
        f"{sorted(unscoped - set(UNSCOPED_EXTERNAL_TOOLS))}; "
        "listed but no longer unscoped: "
        f"{sorted(set(UNSCOPED_EXTERNAL_TOOLS) - unscoped)}"
    )
