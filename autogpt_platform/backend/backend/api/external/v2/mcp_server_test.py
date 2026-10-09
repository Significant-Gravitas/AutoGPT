"""Keep the MCP tool surface deliberate: every Copilot tool is classified."""

from types import SimpleNamespace
from typing import Any
from unittest import mock
from urllib.parse import urlparse

import pytest
import pytest_mock
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest
from prisma.enums import APIKeyPermission
from starlette.routing import Match

from backend.api.external.v2.mcp_server import (
    EXTERNAL_USE_EXCLUSIONS,
    META_KEY_REQUIRED_SCOPES,
    UNSCOPED_EXTERNAL_TOOLS,
    WELL_KNOWN_PROTECTED_RESOURCE_PATH,
    TenantedAccessToken,
    _create_tool_handler,
    create_mcp_server,
    protected_resource_metadata,
)
from backend.copilot.gate.classifier import Judgement
from backend.copilot.model import ChatSession
from backend.copilot.rate_limit import RateLimitUnavailable
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.models import ErrorResponse


def _token(*scopes: APIKeyPermission) -> TenantedAccessToken:
    return TenantedAccessToken(
        token="agpt_test",
        client_id="user-1",
        scopes=[s.value for s in scopes],
        organization_id="org-1",
        team_id="team-1",
    )


def _tool_scopes(tool_name: str) -> list[APIKeyPermission]:
    return list(TOOL_REGISTRY[tool_name].allow_external_use[1] or [])


async def _call(tool_name: str, arguments: dict[str, Any]):
    server = create_mcp_server()
    result = await server._mcp_server.request_handlers[CallToolRequest](
        CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=arguments),
        )
    )
    return result.root


@pytest.fixture
def no_tenancy_lookups(mocker: pytest_mock.MockerFixture) -> mock.AsyncMock:
    """Calls whose ids are all in the caller's organization."""
    return mocker.patch(
        "backend.api.external.v2.mcp_server.check_ids_in_tenant",
        new_callable=mock.AsyncMock,
    )


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


async def test_the_advertised_schema_names_required_arguments_and_nothing_extra():
    registered = {t.name: t for t in await FastMCP.list_tools(create_mcp_server())}

    move = registered["move_agents_to_folder"].inputSchema
    assert move["additionalProperties"] is False
    assert move["required"] == ["agent_ids"]

    # Presets have no v2 permission, and the sandbox paths no sandbox.
    assert "preset_id" not in registered["run_agent"].inputSchema["properties"]
    assert "save_to_path" not in (
        registered["read_workspace_file"].inputSchema["properties"]
    )
    assert "source_path" not in (
        registered["write_workspace_file"].inputSchema["properties"]
    )


def test_create_feature_request_stays_off_the_external_surface():
    """It writes to the platform's own Linear workspace on the platform's key."""
    assert not TOOL_REGISTRY["create_feature_request"].allow_external_use[0]
    assert "create_feature_request" in EXTERNAL_USE_EXCLUSIONS


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
    handler = _create_tool_handler(TOOL_REGISTRY["list_folders"], ["READ_GRAPH"])

    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token", return_value=None
    )
    with pytest.raises(ToolError, match="Authentication required"):
        await handler(ctx=mock.Mock())

    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=_token(APIKeyPermission.READ_RUN),
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


@pytest.mark.asyncio
async def test_tools_list_leaves_out_tools_the_caller_has_no_scope_for(
    mocker: pytest_mock.MockerFixture,
):
    granted = {APIKeyPermission.READ_LIBRARY}
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=SimpleNamespace(scopes=[p.value for p in granted]),
    )
    exposed = {
        name: set(perms)
        for name, t in TOOL_REGISTRY.items()
        for allowed, perms in [t.allow_external_use]
        if allowed
    }
    server = create_mcp_server()

    result = await server._mcp_server.request_handlers[ListToolsRequest](
        ListToolsRequest(method="tools/list")
    )

    listed = {t.name for t in result.root.tools}
    assert listed == {name for name, perms in exposed.items() if perms <= granted}
    assert listed < set(exposed)


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


def test_no_tool_that_spends_platform_money_is_unscoped():
    for name, tool in TOOL_REGISTRY.items():
        allowed, perms = tool.allow_external_use
        if allowed and tool.spends_platform_money:
            assert perms, f"{name} spends platform money but needs no permission"


@pytest.mark.parametrize(
    "token, refusal",
    [
        (None, "Authentication required"),
        # Without a resolved organization the call couldn't be confined to one.
        (
            AccessToken(
                token="agpt_test", client_id="user-1", scopes=["WRITE_LIBRARY"]
            ),
            "Authentication required",
        ),
        (_token(), "Missing required permission"),
    ],
)
async def test_a_call_without_the_right_credentials_reaches_the_client_as_an_error(
    mocker: pytest_mock.MockerFixture, token, refusal: str
):
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token", return_value=token
    )
    run = mocker.patch.object(type(TOOL_REGISTRY["delete_folder"]), "_execute")

    result = await _call("delete_folder", {"folder_id": "folder-1"})

    run.assert_not_called()
    assert result.isError
    assert refusal in result.content[0].text


@pytest.mark.parametrize(
    "arguments, unknown",
    [
        # Presets have no v2 permission.
        ({"library_agent_id": "agent-1", "preset_id": "preset-1"}, "preset_id"),
        # Not in the schema at all; the tool's own input model has the field.
        ({"library_agent_id": "agent-1", "save_as_preset": True}, "save_as_preset"),
        # The marker the approval gate passes to a tool it already approved.
        ({"library_agent_id": "agent-1", "_gate_approved": True}, "_gate_approved"),
    ],
)
async def test_an_argument_the_tool_does_not_advertise_is_refused(
    mocker: pytest_mock.MockerFixture,
    no_tenancy_lookups: mock.AsyncMock,
    arguments: dict,
    unknown: str,
):
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=_token(*_tool_scopes("run_agent")),
    )
    run = mocker.patch.object(type(TOOL_REGISTRY["run_agent"]), "_execute")

    result = await _call("run_agent", arguments)

    run.assert_not_called()
    assert result.isError
    assert f"Unknown argument(s): {unknown}" in result.content[0].text


@pytest.mark.parametrize(
    "arguments, missing",
    [
        ({"library_agent_id": "agent-1", "cron": "0 9 * * *"}, "WRITE_SCHEDULE"),
        ({"library_agent_id": "agent-1", "schedule_name": "daily"}, "WRITE_SCHEDULE"),
        ({"username_agent_slug": "someone/agent"}, "WRITE_LIBRARY"),
    ],
)
async def test_a_branch_needs_the_permission_for_what_it_does(
    mocker: pytest_mock.MockerFixture,
    no_tenancy_lookups: mock.AsyncMock,
    arguments: dict,
    missing: str,
):
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=_token(APIKeyPermission.RUN_AGENT),
    )
    run = mocker.patch.object(type(TOOL_REGISTRY["run_agent"]), "_execute")

    result = await _call("run_agent", arguments)

    run.assert_not_called()
    assert result.isError
    assert f"Missing required permission(s): {missing}" in result.content[0].text


async def test_an_id_in_another_organization_stops_the_call_before_the_tool(
    mocker: pytest_mock.MockerFixture,
):
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=_token(APIKeyPermission.WRITE_LIBRARY),
    )
    library = mocker.patch("backend.api.external.v2.mcp_tenancy.library_db")
    library.return_value.get_folder = mock.AsyncMock(
        return_value=SimpleNamespace(organization_id="org-2")
    )
    run = mocker.patch.object(type(TOOL_REGISTRY["delete_folder"]), "_execute")

    result = await _call("delete_folder", {"folder_id": "folder-1"})

    run.assert_not_called()
    assert result.isError
    assert "Folder 'folder-1' not found" in result.content[0].text
    library.return_value.get_folder.assert_awaited_once_with("folder-1", "user-1")


@pytest.mark.parametrize(
    "allowance_failure, refusal",
    [
        ({"paywalled": True}, "needs an active subscription"),
        (
            {"rate_limit": RateLimitUnavailable("redis down")},
            "Usage limits are unavailable",
        ),
    ],
)
async def test_a_tool_that_spends_platform_money_checks_the_allowance_first(
    mocker: pytest_mock.MockerFixture, allowance_failure: dict, refusal: str
):
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=_token(APIKeyPermission.USE_TOOLS),
    )
    mocker.patch(
        "backend.api.external.v2.mcp_calls.is_user_paywalled",
        new_callable=mock.AsyncMock,
        return_value=allowance_failure.get("paywalled", False),
    )
    mocker.patch(
        "backend.api.external.v2.mcp_calls.get_global_rate_limits",
        new_callable=mock.AsyncMock,
        return_value=(1, 1, None),
    )
    mocker.patch(
        "backend.api.external.v2.mcp_calls.check_rate_limit",
        new_callable=mock.AsyncMock,
        side_effect=allowance_failure.get("rate_limit"),
    )
    run = mocker.patch.object(type(TOOL_REGISTRY["web_search"]), "_execute")

    result = await _call("web_search", {"query": "weather"})

    run.assert_not_called()
    assert result.isError
    assert refusal in result.content[0].text


async def test_a_tool_called_over_mcp_runs_without_waiting_for_an_approval(
    mocker: pytest_mock.MockerFixture, no_tenancy_lookups: mock.AsyncMock
):
    """Nobody is watching an MCP call to answer an approval card.

    The session is an automation's, as a scheduled turn's is, so the gate stays
    out of it and the credential's scopes are the authorization, as in REST.
    """
    tool = TOOL_REGISTRY["delete_folder"]
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=_token(*_tool_scopes("delete_folder")),
    )
    mocker.patch("backend.copilot.gate.is_feature_enabled", return_value=True)
    mocker.patch(
        "backend.copilot.gate.supervise",
        return_value=Judgement(allowed=False, reason="Nobody asked for it."),
    )
    open_review = mocker.patch(
        "backend.copilot.gate.review_store.open_review", return_value=True
    )
    run = mocker.patch.object(
        type(tool),
        "_execute",
        new_callable=mock.AsyncMock,
        return_value=ErrorResponse(message="stub", session_id=None),
    )

    await _call("delete_folder", {"folder_id": "folder-1"})

    open_review.assert_not_called()
    run.assert_awaited_once()
    session = run.await_args.args[1]
    assert isinstance(session, ChatSession)
    assert session.metadata.origin == "automation"
    assert session.external_caller
    assert (session.organization_id, session.team_id) == ("org-1", "team-1")
    no_tenancy_lookups.assert_awaited_once_with(
        "delete_folder", {"folder_id": "folder-1"}, "user-1", "org-1"
    )
