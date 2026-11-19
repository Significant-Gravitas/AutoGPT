"""An MCP call can't name a row in another organization than its credential's."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from mcp.server.fastmcp.exceptions import ToolError

from backend.api.external.v2.mcp_tenancy import (
    TENANTED_ARGS,
    UNTENANTED_ARGS,
    check_ids_in_tenant,
)
from backend.copilot.tools import TOOL_REGISTRY
from backend.util.exceptions import NotFoundError
from backend.util.text_limits import MAX_LIST_ENTRIES

USER = "user-1"
ORG = "org-1"


def _looks_like_an_id(argument: str) -> bool:
    return argument.endswith(("_id", "_ids", "_slug"))


def _exposed_tools() -> dict[str, set[str]]:
    return {
        name: set(tool.external_parameters.get("properties", {}))
        for name, tool in TOOL_REGISTRY.items()
        if tool.allow_external_use[0]
    }


def test_every_id_an_exposed_tool_takes_is_classified():
    unclassified = sorted(
        f"{name}.{arg}"
        for name, arguments in _exposed_tools().items()
        for arg in arguments
        if _looks_like_an_id(arg)
        and arg not in TENANTED_ARGS.get(name, {})
        and arg not in UNTENANTED_ARGS.get(name, {})
    )
    assert not unclassified, (
        "Id arguments neither checked against the organization (TENANTED_ARGS) "
        f"nor listed with a reason (UNTENANTED_ARGS): {unclassified}"
    )


def test_the_classification_names_no_stale_tool_or_argument():
    exposed = _exposed_tools()
    stale = sorted(
        f"{name}.{arg}"
        for table in (TENANTED_ARGS, UNTENANTED_ARGS)
        for name, arguments in table.items()
        for arg in arguments
        if arg not in exposed.get(name, set())
    )
    assert not stale, f"Classified arguments no exposed tool takes: {stale}"


def test_an_argument_named_for_a_list_is_the_tools_only_list():
    """The check refuses a list where a tool takes one id, by the name alone."""
    mismatched = sorted(
        f"{name}.{arg}"
        for name, arguments in TENANTED_ARGS.items()
        for arg in arguments
        if (
            TOOL_REGISTRY[name].external_parameters["properties"][arg].get("type")
            == "array"
        )
        != arg.endswith("_ids")
    )
    assert (
        not mismatched
    ), f"List arguments not named *_ids, or the reverse: {mismatched}"


@pytest.fixture
def library(mocker: pytest_mock.MockerFixture) -> MagicMock:
    lib = MagicMock()
    lib.get_library_agent_organizations_by_any_id = AsyncMock(return_value={})
    lib.get_folder = AsyncMock(side_effect=NotFoundError("no"))
    mocker.patch("backend.api.external.v2.mcp_tenancy.library_db", return_value=lib)
    return lib


@pytest.fixture
def graphs(mocker: pytest_mock.MockerFixture) -> MagicMock:
    db = MagicMock()
    db.get_owned_graph_organizations = AsyncMock(return_value={})
    mocker.patch("backend.api.external.v2.mcp_tenancy.graph_db", return_value=db)
    return db


@pytest.fixture
def runs(mocker: pytest_mock.MockerFixture) -> MagicMock:
    db = MagicMock()
    db.get_graph_execution_meta = AsyncMock(return_value=None)
    mocker.patch("backend.api.external.v2.mcp_tenancy.execution_db", return_value=db)
    return db


def _row(organization_id: str | None, **fields) -> SimpleNamespace:
    return SimpleNamespace(organization_id=organization_id, **fields)


async def test_a_folder_in_another_organization_is_not_found(library: MagicMock):
    library.get_folder = AsyncMock(return_value=_row("org-2"))

    with pytest.raises(ToolError, match="Folder 'folder-1' not found"):
        await check_ids_in_tenant("delete_folder", {"folder_id": "folder-1"}, USER, ORG)


@pytest.mark.parametrize("organization_id", [ORG, None])
async def test_a_folder_in_the_organization_or_untagged_passes(
    library: MagicMock, organization_id: str | None
):
    library.get_folder = AsyncMock(return_value=_row(organization_id))

    await check_ids_in_tenant("delete_folder", {"folder_id": "folder-1"}, USER, ORG)


@pytest.mark.parametrize("tool", ["update_preset", "delete_preset"])
async def test_a_preset_in_another_organization_is_not_found(
    library: MagicMock, tool: str
):
    library.get_preset = AsyncMock(return_value=_row("org-2"))

    with pytest.raises(ToolError, match="Preset 'preset-1' not found"):
        await check_ids_in_tenant(tool, {"preset_id": "preset-1"}, USER, ORG)
    library.get_preset.assert_awaited_once_with(user_id=USER, preset_id="preset-1")


async def test_an_id_naming_nothing_is_left_to_the_tool(
    library: MagicMock, graphs: MagicMock, runs: MagicMock
):
    await check_ids_in_tenant(
        "view_agent_output",
        {"library_agent_id": "agent-1", "execution_id": "run-1"},
        USER,
        ORG,
    )


async def test_a_library_agent_in_another_organization_is_not_found(
    library: MagicMock, graphs: MagicMock
):
    library.get_library_agent_organizations_by_any_id.return_value = {
        "agent-1": ["org-2"]
    }

    with pytest.raises(ToolError, match="Library agent 'agent-1' not found"):
        await check_ids_in_tenant(
            "run_agent", {"library_agent_id": "agent-1"}, USER, ORG
        )
    graphs.get_owned_graph_organizations.assert_not_called()


async def test_an_id_naming_two_entries_needs_both_in_the_organization(
    library: MagicMock, graphs: MagicMock
):
    """One entry by its own id, another by its graph's."""
    library.get_library_agent_organizations_by_any_id.return_value = {
        "graph-1": [ORG, "org-2"]
    }

    with pytest.raises(ToolError, match="Library agent 'graph-1' not found"):
        await check_ids_in_tenant("edit_agent", {"agent_id": "graph-1"}, USER, ORG)


async def test_an_own_graph_without_a_library_entry_is_checked_by_its_own_tag(
    library: MagicMock, graphs: MagicMock
):
    graphs.get_owned_graph_organizations.return_value = {"graph-1": "org-2"}

    with pytest.raises(ToolError, match="Library agent 'graph-1' not found"):
        await check_ids_in_tenant("edit_agent", {"agent_id": "graph-1"}, USER, ORG)
    graphs.get_owned_graph_organizations.assert_awaited_once_with(USER, ["graph-1"])


async def test_a_list_is_resolved_in_one_read_and_each_id_once(
    library: MagicMock, graphs: MagicMock
):
    ids = [f"agent-{i}" for i in range(MAX_LIST_ENTRIES - 1)] + [" agent-0 "]
    unique = ids[:-1]
    library.get_library_agent_organizations_by_any_id.return_value = {"agent-0": [ORG]}

    await check_ids_in_tenant("move_agents_to_folder", {"agent_ids": ids}, USER, ORG)

    library.get_library_agent_organizations_by_any_id.assert_awaited_once_with(
        USER, unique
    )
    graphs.get_owned_graph_organizations.assert_awaited_once_with(USER, unique[1:])


@pytest.mark.parametrize(
    "tool, args",
    [
        ("move_agents_to_folder", {"agent_ids": ["agent-1"] * (MAX_LIST_ENTRIES + 1)}),
        ("create_agent", {"library_agent_ids": "agent-1"}),
        ("create_agent", {"library_agent_ids": {"agent-1": "agent-2"}}),
        ("move_agents_to_folder", {"agent_ids": ["agent-1", 7]}),
        ("list_presets", {"graph_id": ["graph-1", "graph-2"]}),
        ("delete_folder", {"folder_id": {"id": "folder-1"}}),
    ],
)
async def test_an_argument_the_tool_cant_take_is_refused_unread(
    library: MagicMock, graphs: MagicMock, tool: str, args: dict
):
    with pytest.raises(ToolError, match="must be|at most"):
        await check_ids_in_tenant(tool, args, USER, ORG)

    library.get_library_agent_organizations_by_any_id.assert_not_called()
    library.get_folder.assert_not_called()
    graphs.get_owned_graph_organizations.assert_not_called()


async def test_every_id_in_a_list_is_checked(library: MagicMock, graphs: MagicMock):
    library.get_library_agent_organizations_by_any_id.return_value = {
        "agent-1": [ORG],
        "agent-2": ["org-2"],
    }

    with pytest.raises(ToolError, match="Library agent 'agent-2' not found"):
        await check_ids_in_tenant(
            "move_agents_to_folder",
            {"agent_ids": ["agent-1", "agent-2"]},
            USER,
            ORG,
        )


async def test_an_id_is_checked_the_way_the_tool_reads_it(library: MagicMock):
    """The tools strip the ids they're given, so the check does too."""
    library.get_folder = AsyncMock(return_value=_row("org-2"))

    with pytest.raises(ToolError):
        await check_ids_in_tenant(
            "delete_folder", {"folder_id": "  folder-1 "}, USER, ORG
        )
    library.get_folder.assert_awaited_once_with("folder-1", USER)


async def test_a_run_in_another_organization_is_not_found(runs: MagicMock):
    runs.get_graph_execution_meta = AsyncMock(return_value=_row("org-2"))

    with pytest.raises(ToolError, match="Run 'run-1' not found"):
        await check_ids_in_tenant(
            "view_agent_output", {"execution_id": "run-1"}, USER, ORG
        )
    runs.get_graph_execution_meta.assert_awaited_once_with(
        user_id=USER, execution_id="run-1"
    )


async def test_a_failed_lookup_refuses_without_leaking_the_error(library: MagicMock):
    library.get_folder = AsyncMock(side_effect=RuntimeError("query engine down"))

    with pytest.raises(ToolError) as refusal:
        await check_ids_in_tenant("delete_folder", {"folder_id": "folder-1"}, USER, ORG)
    assert "query engine" not in str(refusal.value)


async def test_a_tool_without_ids_needs_no_lookup(library: MagicMock):
    await check_ids_in_tenant("search_docs", {"query": "folder_id"}, USER, ORG)

    library.get_folder.assert_not_called()
