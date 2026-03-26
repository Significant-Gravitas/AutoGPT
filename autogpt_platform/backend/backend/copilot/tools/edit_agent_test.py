"""Tests for EditAgentTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.edit_agent import EditAgentTool
from backend.copilot.tools.models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    ErrorResponse,
)

from ._test_data import make_session

_TEST_USER_ID = "test-user-edit-agent"
_PIPELINE = "backend.copilot.tools.agent_generator.pipeline"
_EDIT_AGENT = "backend.copilot.tools.edit_agent"


def _make_agent_json(
    *,
    agent_id: str = "graph-123",
    version: int = 1,
    name: str = "Existing Agent",
) -> dict:
    return {
        "id": agent_id,
        "version": version,
        "name": name,
        "description": "An existing agent",
        "is_active": True,
        "nodes": [
            {
                "id": "node-1",
                "block_id": "block-1",
                "input_default": {},
                "metadata": {"position": {"x": 0, "y": 0}},
            }
        ],
        "links": [],
    }


@pytest.fixture
def tool():
    return EditAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── Input validation tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_agent_id_returns_error(tool, session):
    """Missing agent_id returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_json=_make_agent_json(),
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_id"


@pytest.mark.asyncio
async def test_missing_agent_json_returns_error(tool, session):
    """Missing agent_json returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_id="graph-123",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_json"


@pytest.mark.asyncio
async def test_empty_nodes_returns_error(tool, session):
    """agent_json with no nodes returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_id="graph-123",
        agent_json={"nodes": [], "links": []},
    )
    assert isinstance(result, ErrorResponse)
    assert "no nodes" in result.message.lower()


@pytest.mark.asyncio
async def test_agent_not_found_returns_error(tool, session):
    """Non-existent agent_id returns ErrorResponse."""
    with patch(
        f"{_EDIT_AGENT}.get_agent_as_json",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="nonexistent-id",
            agent_json=_make_agent_json(),
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "agent_not_found"


# ── Preview mode tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preview_mode_returns_preview(tool, session):
    """save=False with valid agent_json returns AgentPreviewResponse."""
    existing_agent = _make_agent_json()
    updated_json = _make_agent_json(name="Updated Agent")

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = MagicMock(return_value=updated_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    with (
        patch(
            f"{_EDIT_AGENT}.get_agent_as_json",
            new_callable=AsyncMock,
            return_value=existing_agent,
        ),
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="graph-123",
            agent_json=updated_json,
            save=False,
        )

    assert isinstance(result, AgentPreviewResponse)
    assert result.agent_name == "Updated Agent"
    assert result.node_count == 1


@pytest.mark.asyncio
async def test_no_auth_returns_error(tool, session):
    """save=True without user_id returns ErrorResponse about login."""
    existing_agent = _make_agent_json()
    updated_json = _make_agent_json(name="Updated Agent")

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = MagicMock(return_value=updated_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    with (
        patch(
            f"{_EDIT_AGENT}.get_agent_as_json",
            new_callable=AsyncMock,
            return_value=existing_agent,
        ),
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
    ):
        result = await tool._execute(
            user_id=None,
            session=session,
            agent_id="graph-123",
            agent_json=updated_json,
            save=True,
        )

    assert isinstance(result, ErrorResponse)
    assert "logged in" in result.message.lower()


# ── Save tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_calls_pipeline_with_is_update_true(tool, session):
    """Saving passes is_update=True and preserves the original agent's id."""
    existing_agent = _make_agent_json(agent_id="original-graph-id", version=3)
    updated_json = _make_agent_json(name="Edited Agent")

    mock_graph = MagicMock()
    mock_graph.id = "original-graph-id"
    mock_graph.name = "Edited Agent"
    mock_graph.version = 4

    mock_library_agent = MagicMock()
    mock_library_agent.id = "lib-agent-1"

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = MagicMock(return_value=updated_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    mock_save = AsyncMock(return_value=(mock_graph, mock_library_agent))

    with (
        patch(
            f"{_EDIT_AGENT}.get_agent_as_json",
            new_callable=AsyncMock,
            return_value=existing_agent,
        ),
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
        patch(f"{_PIPELINE}.save_agent_to_library", mock_save),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="original-graph-id",
            agent_json=updated_json,
            save=True,
        )

    assert isinstance(result, AgentSavedResponse)
    assert result.agent_name == "Edited Agent"

    # Verify save_agent_to_library was called with is_update=True
    mock_save.assert_called_once()
    call_kwargs = mock_save.call_args
    assert call_kwargs[1].get("is_update") or call_kwargs[0][2] is True
    # The agent_json passed to save should have the original graph ID
    saved_json = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]["agent_json"]
    assert saved_json["id"] == "original-graph-id"


@pytest.mark.asyncio
async def test_save_preserves_original_version(tool, session):
    """Edit sets agent_json version to the current agent's version."""
    existing_agent = _make_agent_json(agent_id="graph-abc", version=5)
    updated_json = _make_agent_json(name="Edited v5")

    mock_graph = MagicMock()
    mock_graph.id = "graph-abc"
    mock_graph.name = "Edited v5"
    mock_graph.version = 6

    mock_library_agent = MagicMock()
    mock_library_agent.id = "lib-1"

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = MagicMock(return_value=updated_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    mock_save = AsyncMock(return_value=(mock_graph, mock_library_agent))

    with (
        patch(
            f"{_EDIT_AGENT}.get_agent_as_json",
            new_callable=AsyncMock,
            return_value=existing_agent,
        ),
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
        patch(f"{_PIPELINE}.save_agent_to_library", mock_save),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="graph-abc",
            agent_json=updated_json,
            save=True,
        )

    assert isinstance(result, AgentSavedResponse)
    saved_json = mock_save.call_args[0][0]
    assert saved_json["version"] == 5


@pytest.mark.asyncio
async def test_save_failure_returns_error(tool, session):
    """When save_agent_to_library raises, returns ErrorResponse."""
    existing_agent = _make_agent_json()
    updated_json = _make_agent_json(name="Will Fail")

    mock_fixer = MagicMock()
    mock_fixer.apply_all_fixes = MagicMock(return_value=updated_json)
    mock_fixer.get_fixes_applied.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate.return_value = (True, None)
    mock_validator.errors = []

    mock_save = AsyncMock(
        side_effect=Exception(
            "Unique constraint failed on the fields: (`id`,`version`)"
        )
    )

    with (
        patch(
            f"{_EDIT_AGENT}.get_agent_as_json",
            new_callable=AsyncMock,
            return_value=existing_agent,
        ),
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=mock_fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=mock_validator),
        patch(f"{_PIPELINE}.save_agent_to_library", mock_save),
    ):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_id="graph-123",
            agent_json=updated_json,
            save=True,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "save_failed"
    assert "Unique constraint" in result.message


# ── Bug reproduction: version conflict (xfail) ──────────────────────────


@pytest.mark.xfail(
    reason="SECRT-2175: update_graph_in_library version lookup misses "
    "non-owned graphs, causing unique constraint violation on (id, version)",
    strict=True,
)
@pytest.mark.asyncio
async def test_edit_agent_version_conflict_when_user_not_graph_owner(tool, session):
    """Editing an agent whose graph is owned by a different user should
    still succeed (version should auto-increment correctly).

    Currently fails because update_graph_in_library calls
    get_graph_all_versions(graph_id, user_id) which filters by userId.
    If the editing user is not the graph owner (e.g. store-forked agent),
    the version lookup returns empty → version defaults to 1 → unique
    constraint violation since version 1 already exists.

    This test exercises the real update_graph_in_library logic with mocked
    Prisma calls to reproduce the exact production failure:
    1. get_graph_all_versions returns empty (user doesn't own the graph)
    2. Version defaults to 1
    3. create_graph hits unique constraint because version 1 already exists
    """
    from backend.api.features.library.db import update_graph_in_library
    from backend.data.graph import Graph, Node

    graph = Graph(
        id="shared-graph-id",
        version=1,
        name="Citedy Topic to Publish",
        description="An agent owned by another user",
        is_active=True,
        nodes=[
            Node(
                id="node-1",
                block_id="c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
                input_default={},
                metadata={"position": {"x": 0, "y": 0}},
            )
        ],
        links=[],
    )

    # Mock get_graph_all_versions to return empty (simulates userId mismatch)
    # and create_graph to raise the unique constraint violation
    with (
        patch(
            "backend.api.features.library.db.graph_db.get_graph_all_versions",
            new_callable=AsyncMock,
            return_value=[],  # Empty: user doesn't own this graph
        ),
        patch(
            "backend.api.features.library.db.graph_db.make_graph_model",
            return_value=graph,
        ),
        patch(
            "backend.api.features.library.db.graph_db.create_graph",
            new_callable=AsyncMock,
        ) as mock_create,
    ):
        # With existing_versions empty, version will be set to 1.
        # The real DB would reject this with UniqueViolationError since
        # version 1 already exists. We verify the version passed to
        # create_graph is > 1 (which it won't be — that's the bug).
        mock_create.return_value = graph
        # Need to mock the library agent lookup too
        with (
            patch(
                "backend.api.features.library.db.get_library_agent_by_graph_id",
                new_callable=AsyncMock,
                return_value=MagicMock(id="lib-1"),
            ),
            patch(
                "backend.api.features.library.db.update_library_agent_version_and_settings",
                new_callable=AsyncMock,
                return_value=MagicMock(id="lib-1"),
            ),
            patch(
                "backend.api.features.library.db.on_graph_activate",
                new_callable=AsyncMock,
                return_value=graph,
            ),
            patch(
                "backend.api.features.library.db.graph_db.set_graph_active_version",
                new_callable=AsyncMock,
            ),
        ):
            await update_graph_in_library(graph, _TEST_USER_ID)

            # The bug: with empty existing_versions, version defaults to 1
            # which collides with the already-existing version 1.
            # This assertion verifies the version was incremented above 1.
            created_graph_arg = mock_create.call_args[0][0]
            assert created_graph_arg.version > 1, (
                f"Expected version > 1 but got {created_graph_arg.version}. "
                "update_graph_in_library defaults to version=1 when "
                "get_graph_all_versions returns empty (userId mismatch), "
                "causing unique constraint violation on (id, version)."
            )
