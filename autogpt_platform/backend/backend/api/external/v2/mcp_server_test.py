"""Keep the MCP tool surface deliberate: every Copilot tool is classified."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest_mock

from backend.api.external.v2.mcp_server import (
    EXTERNAL_USE_EXCLUSIONS,
    UNSCOPED_EXTERNAL_TOOLS,
    _create_tool_handler,
)
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


async def test_a_gated_tool_called_over_mcp_is_parked_for_an_approval_nobody_can_give(
    mocker: pytest_mock.MockerFixture,
):
    # Pins today's behaviour, pending a design decision: the MCP session is
    # "interactive", so under copilot-auto-mode the gate parks the call.
    tool = TOOL_REGISTRY["delete_folder"]
    _, scopes = tool.allow_external_use
    mocker.patch(
        "backend.api.external.v2.mcp_server.get_access_token",
        return_value=SimpleNamespace(
            client_id="user-1", scopes=[str(s) for s in scopes or []]
        ),
    )
    mocker.patch("backend.copilot.gate.is_feature_enabled", return_value=True)
    mocker.patch("backend.copilot.gate.chat_rules.ask_reason", return_value=None)
    mocker.patch("backend.copilot.gate.review_store.find_decision", return_value=None)
    mocker.patch(
        "backend.copilot.gate.classify", return_value=(False, "Nobody asked for it.")
    )
    mocker.patch("backend.copilot.gate.held.remember", return_value=True)
    open_review = mocker.patch(
        "backend.copilot.gate.review_store.open_review", return_value=True
    )
    run = mocker.patch.object(type(tool), "_execute")

    output = await _create_tool_handler(tool, [str(s) for s in scopes or []])(
        ctx=MagicMock(), folder_id="folder-1"
    )

    run.assert_not_called()
    open_review.assert_awaited_once()
    assert json.loads(output)["type"] == "approval_required"
