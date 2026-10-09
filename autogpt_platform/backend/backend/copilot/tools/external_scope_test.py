"""Copilot tools called through the External API stay inside its organization.

The MCP server checks the ids a call names (`api/external/v2/mcp_tenancy.py`);
these pin what the tools find, list and create on their own.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import APIKeyPermission

from backend.copilot.model import ChatSession
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.agent_output import AgentOutputTool, _run_visible
from backend.copilot.tools.agent_search import (
    _get_library_agent_by_id,
    _load_and_format_matched_agents,
    search_agents,
)
from backend.copilot.tools.create_agent import CreateAgentTool
from backend.copilot.tools.external_scope import (
    external_tenancy,
    external_tenant,
    in_tenant,
)
from backend.copilot.tools.helpers import require_guide_read, require_library_check
from backend.copilot.tools.manage_folders import ListFoldersTool
from backend.copilot.tools.manage_schedules import _is_in_session_scope
from backend.copilot.tools.models import ErrorResponse

USER = "user-1"


def _external(organization_id: str = "org-1") -> ChatSession:
    session = ChatSession.new(
        USER,
        dry_run=False,
        origin="automation",
        organization_id=organization_id,
        team_id="team-1",
    )
    session.external_caller = True
    return session


def _chat(organization_id: str = "org-1") -> ChatSession:
    return ChatSession.new(USER, dry_run=False, organization_id=organization_id)


def test_only_an_external_call_is_confined():
    assert external_tenant(_external()) == "org-1"
    assert external_tenancy(_external()) == ("org-1", "team-1")
    # A chat keeps every behaviour it had: the web app owns its org context.
    assert external_tenant(_chat()) is None
    assert external_tenancy(_chat()) == (None, None)


@pytest.mark.parametrize(
    "organization_id, tenant, visible",
    [
        ("org-1", "org-1", True),
        ("org-2", "org-1", False),
        # Untagged rows predate tagging and stay visible to their owner.
        (None, "org-1", True),
        ("org-2", None, True),
    ],
)
def test_in_tenant(organization_id: str | None, tenant: str | None, visible: bool):
    assert in_tenant(organization_id, tenant) is visible


def test_an_external_call_needs_no_earlier_call_in_its_session():
    """Each MCP call gets a fresh session, so a history check could never pass."""
    assert require_guide_read(_external(), "create_agent") is None
    assert require_library_check(_external(), "create_agent") is None

    # The flag is what lets it through: the same fresh session in a chat asks.
    assert isinstance(require_guide_read(_chat(), "create_agent"), ErrorResponse)
    assert isinstance(require_library_check(_chat(), "create_agent"), ErrorResponse)


@pytest.mark.parametrize(
    "tool_name, args, needed",
    [
        ("create_agent", {"agent_json": {"nodes": [{"id": "n1"}]}}, []),
        (
            "create_agent",
            {"agent_json_ref": "workspace:///agent.json"},
            [APIKeyPermission.READ_FILES],
        ),
        # A string that isn't inline JSON is read as a file reference too.
        (
            "create_agent",
            {"agent_json": "@@agptfile:workspace:///agent.json"},
            [APIKeyPermission.READ_FILES],
        ),
        (
            "customize_agent",
            {"agent_json": {"nodes": [{"id": "n1"}]}, "library_agent_ids": ["a"]},
            [APIKeyPermission.READ_LIBRARY],
        ),
        (
            "edit_agent",
            {"agent_json_ref": "agent.json", "library_agent_ids": ["a"]},
            [APIKeyPermission.READ_FILES, APIKeyPermission.READ_LIBRARY],
        ),
        (
            "validate_agent_graph",
            {"agent_json_ref": "agent.json"},
            [APIKeyPermission.READ_FILES],
        ),
        (
            "fix_agent_graph",
            {"agent_json": {"nodes": [{"id": "n1"}]}, "write_to": "fixed.json"},
            [APIKeyPermission.WRITE_FILES],
        ),
        (
            "find_library_agent",
            {"agent_id": "a", "write_graph_to": "graph.json"},
            [APIKeyPermission.WRITE_FILES],
        ),
        ("run_agent", {"library_agent_id": "a"}, []),
        (
            "run_agent",
            {"library_agent_id": "a", "cron": "0 9 * * *"},
            [APIKeyPermission.WRITE_SCHEDULE],
        ),
        (
            "run_agent",
            {"username_agent_slug": "someone/agent"},
            [APIKeyPermission.WRITE_LIBRARY],
        ),
    ],
)
def test_arguments_add_the_permission_for_what_they_make_the_tool_do(
    tool_name: str, args: dict[str, Any], needed: list[APIKeyPermission]
):
    assert TOOL_REGISTRY[tool_name].external_permissions(args) == needed


async def test_a_library_search_lists_only_the_organizations_agents():
    lib = MagicMock()
    lib.list_library_agents = AsyncMock(return_value=SimpleNamespace(agents=[]))

    with patch("backend.copilot.tools.agent_search.library_db", return_value=lib):
        await search_agents(
            query="emails",
            source="library",
            session_id="s",
            user_id=USER,
            tenant="org-1",
        )

    assert lib.list_library_agents.await_args.kwargs["organization_id"] == "org-1"


@pytest.mark.parametrize(
    "lookup", ["get_library_agent_by_graph_id", "get_library_agent"]
)
async def test_an_agent_in_another_organization_is_not_found_by_id(lookup: str):
    lib = MagicMock()
    lib.get_library_agent_by_graph_id = AsyncMock(return_value=None)
    lib.get_library_agent = AsyncMock(return_value=None)
    setattr(
        lib, lookup, AsyncMock(return_value=SimpleNamespace(organization_id="org-2"))
    )

    with patch("backend.copilot.tools.agent_search.library_db", return_value=lib):
        assert await _get_library_agent_by_id(USER, "agent-1", "org-1") is None


async def test_a_similarity_match_in_another_organization_is_skipped():
    lib = MagicMock()
    lib.get_library_agent = AsyncMock(
        return_value=SimpleNamespace(organization_id="org-2")
    )

    with patch("backend.copilot.tools.agent_search.library_db", return_value=lib):
        found = await _load_and_format_matched_agents(
            [{"content_id": "agent-1", "combined_score": 0.9}], USER, "org-1"
        )

    assert found == []


@pytest.mark.parametrize(
    "organization_id, visible", [("org-1", True), ("org-2", False), (None, True)]
)
def test_a_run_is_visible_only_in_its_organization(
    organization_id: str | None, visible: bool
):
    run = MagicMock(expert_id=None, organization_id=organization_id)
    assert _run_visible(run, None, "org-1") is visible


async def test_run_outputs_of_an_agent_in_another_organization_are_not_found():
    lib = MagicMock()
    lib.get_library_agent = AsyncMock(
        return_value=SimpleNamespace(organization_id="org-2")
    )

    with patch("backend.copilot.tools.agent_output.library_db", return_value=lib):
        agent, error = await AgentOutputTool()._resolve_agent(
            user_id=USER,
            agent_name=None,
            library_agent_id="agent-1",
            store_slug=None,
            tenant="org-1",
        )

    assert agent is None
    assert error == "Library agent 'agent-1' not found"


@pytest.mark.parametrize(
    "session, organization_id, in_scope",
    [
        (_external(), "org-1", True),
        (_external(), "org-2", False),
        # Graph schedules persisted before tagging carry "".
        (_external(), "", True),
        (_chat(), "org-2", True),
    ],
)
def test_schedule_tools_see_only_the_organizations_schedules(
    session: ChatSession, organization_id: str, in_scope: bool
):
    job = MagicMock(expert_id=None, organization_id=organization_id)
    assert _is_in_session_scope(job, session) is in_scope


async def test_listing_folders_lists_only_the_organizations():
    lib = MagicMock()
    lib.get_folder_tree = AsyncMock(return_value=[])
    lib.get_folder_agents_map = AsyncMock(return_value={})
    lib.get_root_agent_summaries = AsyncMock(return_value=[])

    with patch("backend.copilot.tools.manage_folders.library_db", return_value=lib):
        await ListFoldersTool()._execute(USER, _external(), include_agents=True)

    lib.get_folder_tree.assert_awaited_once_with(user_id=USER, organization_id="org-1")
    lib.get_folder_agents_map.assert_awaited_once_with(USER, [], "org-1")
    lib.get_root_agent_summaries.assert_awaited_once_with(USER, "org-1")


async def test_an_agent_created_over_the_external_api_lands_in_its_organization():
    save = AsyncMock(return_value=ErrorResponse(message="stub", session_id=None))

    with (
        patch("backend.copilot.tools.create_agent.fix_validate_and_save", save),
        patch(
            "backend.copilot.tools.create_agent.fetch_library_agents",
            AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.tools.create_agent.install_saved_agent",
            AsyncMock(side_effect=lambda _user, _session, saved: saved),
        ),
    ):
        await CreateAgentTool()._execute(
            USER, _external(), agent_json={"nodes": [{"id": "n1"}], "links": []}
        )

    assert save.await_args.kwargs["organization_id"] == "org-1"
    assert save.await_args.kwargs["team_id"] == "team-1"
