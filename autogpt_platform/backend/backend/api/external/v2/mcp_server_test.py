"""Keep the MCP tool surface deliberate: every Copilot tool is classified."""

from unittest import mock
from urllib.parse import urlparse

import pytest
import pytest_mock
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from starlette.routing import Match

from backend.api.external.v2.mcp_server import (
    EXTERNAL_USE_EXCLUSIONS,
    META_KEY_REQUIRED_SCOPES,
    UNSCOPED_EXTERNAL_TOOLS,
    WELL_KNOWN_PROTECTED_RESOURCE_PATH,
    _create_tool_handler,
    create_mcp_server,
    protected_resource_metadata,
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


def test_the_protected_resource_document_answers_where_clients_look():
    """RFC 9728 discovery, at the URL the `WWW-Authenticate` header names.

    FastMCP registers its own copy inside the `/mcp` mount, so the derived URL
    — well-known segment first, resource path after — 404s unless the root app
    serves it. This asserts against the real app, not a stand-in.
    """
    from mcp.server.auth.routes import build_resource_metadata_url

    from backend.api.rest_api import app

    metadata = protected_resource_metadata()
    derived = urlparse(str(build_resource_metadata_url(metadata.resource)))
    assert derived.path == WELL_KNOWN_PROTECTED_RESOURCE_PATH

    scope = {"type": "http", "method": "GET", "path": derived.path, "headers": []}
    assert any(
        route.matches(scope)[0] == Match.FULL for route in app.routes
    ), f"nothing on the root app answers {derived.path}"


async def test_a_rejected_tool_call_is_an_error_not_a_successful_denial(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Returned text is reported to the client as a call that succeeded."""
    handler = _create_tool_handler(next(iter(TOOL_REGISTRY.values())), ["READ_GRAPH"])

    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token", return_value=None
    )
    with pytest.raises(ToolError, match="Authentication required"):
        await handler(ctx=mock.Mock())

    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=mock.Mock(scopes=["READ_RUN"]),
    )
    with pytest.raises(ToolError, match="READ_GRAPH"):
        await handler(ctx=mock.Mock())


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
