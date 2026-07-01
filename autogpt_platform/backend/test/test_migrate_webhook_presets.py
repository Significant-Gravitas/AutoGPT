"""
Unit tests for migrate_webhook_presets_to_new_version.
Mocks prisma to avoid needing a running database.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.db import migrate_webhook_presets_to_new_version

# Patch prisma.models.AgentPreset.prisma per the project-wide convention used
# in backend/api/features/library/db_test.py and the rest of the suite.
_PRISMA_PATCH_TARGET = "prisma.models.AgentPreset.prisma"


@pytest.fixture
def mock_prisma():
    with patch(_PRISMA_PATCH_TARGET) as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


def _make_graph(
    *,
    block_id: str = "trigger-block-a",
    has_trigger: bool = True,
    has_config: bool = True,
    graph_id: str = "graph-abc",
    version: int = 5,
):
    """Stand-in for a GraphModel whose trigger node has a block_id + config."""
    graph = MagicMock()
    graph.id = graph_id
    graph.version = version
    if not has_trigger:
        graph.webhook_input_node = None
    elif not has_config:
        # Trigger node present but without a webhook_config (defensive branch).
        graph.webhook_input_node.block.webhook_config = None
    else:
        node = MagicMock()
        node.block_id = block_id
        node.block.webhook_config = MagicMock()  # truthy
        graph.webhook_input_node = node
    return graph


def _make_preset(preset_id: str, *, version: int = 1):
    """Stand-in for a prisma AgentPreset row."""
    preset = MagicMock()
    preset.id = preset_id
    preset.agentGraphVersion = version
    return preset


def _patch_old_graphs(mocker, block_id_by_version: dict[int, str | None]):
    """Patch graph_db.get_graph to return old versions with given trigger blocks.

    A ``None`` value simulates a version that is missing or has no trigger node.
    """

    async def fake_get_graph(graph_id, version, user_id):
        block_id = block_id_by_version.get(version)
        if block_id is None:
            return None
        return _make_graph(block_id=block_id, version=version)

    return mocker.patch(
        "backend.api.features.library.db.graph_db.get_graph",
        side_effect=fake_get_graph,
    )


@pytest.mark.asyncio
async def test_migrate_updates_compatible_presets(mock_prisma, mocker):
    """Presets whose pinned version uses the same trigger block are migrated."""
    graph = _make_graph(block_id="trigger-a", version=5)
    mock_prisma.find_many = AsyncMock(
        return_value=[_make_preset("p1", version=1), _make_preset("p2", version=1)]
    )
    mock_prisma.update_many = AsyncMock(return_value=2)
    _patch_old_graphs(mocker, {1: "trigger-a"})

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 2
    mock_prisma.find_many.assert_called_once_with(
        where={
            "userId": "user-123",
            "agentGraphId": "graph-abc",
            "agentGraphVersion": {"lt": 5},
            "webhookId": {"not": None},
            "isDeleted": False,
        },
    )
    mock_prisma.update_many.assert_called_once_with(
        where={
            "id": {"in": ["p1", "p2"]},
            "userId": "user-123",
            "agentGraphVersion": {"lt": 5},
            "isDeleted": False,
        },
        data={"agentGraphVersion": 5},
    )


@pytest.mark.asyncio
async def test_migrate_skips_different_trigger_provider(mock_prisma, mocker):
    """v1 used a Telegram trigger, v2 uses a GitHub trigger -> do not migrate."""
    graph = _make_graph(block_id="github-on-pr", version=5)
    mock_prisma.find_many = AsyncMock(return_value=[_make_preset("p1", version=1)])
    mock_prisma.update_many = AsyncMock(return_value=0)
    _patch_old_graphs(mocker, {1: "telegram-on-message"})

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 0
    mock_prisma.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_skips_different_trigger_same_provider(mock_prisma, mocker):
    """Same provider, different trigger block (on-PR vs on-issue) -> do not migrate."""
    graph = _make_graph(block_id="github-on-issue", version=5)
    mock_prisma.find_many = AsyncMock(return_value=[_make_preset("p1", version=1)])
    mock_prisma.update_many = AsyncMock(return_value=0)
    _patch_old_graphs(mocker, {1: "github-on-pr"})

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 0
    mock_prisma.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_only_updates_compatible_in_mixed_set(mock_prisma, mocker):
    """Compatible presets migrate; incompatible ones are left pinned."""
    graph = _make_graph(block_id="trigger-a", version=7)
    mock_prisma.find_many = AsyncMock(
        return_value=[
            _make_preset("ok1", version=1),
            _make_preset("bad", version=2),
            _make_preset("ok2", version=1),
        ]
    )
    mock_prisma.update_many = AsyncMock(return_value=2)
    _patch_old_graphs(mocker, {1: "trigger-a", 2: "trigger-b"})

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 2
    mock_prisma.update_many.assert_called_once_with(
        where={
            "id": {"in": ["ok1", "ok2"]},
            "userId": "user-123",
            "agentGraphVersion": {"lt": 7},
            "isDeleted": False,
        },
        data={"agentGraphVersion": 7},
    )


@pytest.mark.asyncio
async def test_migrate_returns_zero_when_no_trigger_node(mock_prisma):
    """No webhook trigger on the new version -> no DB access, returns 0."""
    graph = _make_graph(has_trigger=False)
    mock_prisma.find_many = AsyncMock()
    mock_prisma.update_many = AsyncMock()

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 0
    mock_prisma.find_many.assert_not_called()
    mock_prisma.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_returns_zero_when_trigger_has_no_webhook_config(mock_prisma):
    """Trigger node present but without a webhook_config -> no DB access."""
    graph = _make_graph(has_config=False)
    mock_prisma.find_many = AsyncMock()
    mock_prisma.update_many = AsyncMock()

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 0
    mock_prisma.find_many.assert_not_called()
    mock_prisma.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_skips_preset_when_old_version_unavailable(
    mock_prisma, mocker, caplog
):
    """If the pinned version can't be loaded, treat as incompatible (no downgrade)."""
    graph = _make_graph(block_id="trigger-a", version=5)
    mock_prisma.find_many = AsyncMock(return_value=[_make_preset("orphan", version=1)])
    mock_prisma.update_many = AsyncMock(return_value=0)
    _patch_old_graphs(mocker, {1: None})  # get_graph returns None

    with caplog.at_level(logging.WARNING, logger="backend.api.features.library.db"):
        count = await migrate_webhook_presets_to_new_version(
            user_id="user-123", new_graph=graph
        )

    assert count == 0
    mock_prisma.update_many.assert_not_called()
    assert any(
        "Not migrating preset #orphan" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_migrate_returns_zero_when_no_candidates(mock_prisma, mocker):
    graph = _make_graph(version=3)
    mock_prisma.find_many = AsyncMock(return_value=[])
    mock_prisma.update_many = AsyncMock()
    get_graph_mock = _patch_old_graphs(mocker, {})

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", new_graph=graph
    )

    assert count == 0
    mock_prisma.update_many.assert_not_called()
    get_graph_mock.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_logs_when_presets_are_migrated(mock_prisma, mocker, caplog):
    """Exercise the ``count > 0`` log branch."""
    graph = _make_graph(block_id="trigger-a", version=4)
    mock_prisma.find_many = AsyncMock(
        return_value=[_make_preset("p1", version=1), _make_preset("p2", version=1)]
    )
    mock_prisma.update_many = AsyncMock(return_value=2)
    _patch_old_graphs(mocker, {1: "trigger-a"})

    with caplog.at_level(logging.INFO, logger="backend.api.features.library.db"):
        count = await migrate_webhook_presets_to_new_version(
            user_id="user-789", new_graph=graph
        )

    assert count == 2
    assert any(
        "Migrated 2 webhook preset(s)" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_migrate_warns_on_incompatible_preset(mock_prisma, mocker, caplog):
    """Incompatible presets emit a warning explaining why they were skipped."""
    graph = _make_graph(block_id="github-on-pr", version=5)
    mock_prisma.find_many = AsyncMock(return_value=[_make_preset("bad", version=1)])
    mock_prisma.update_many = AsyncMock(return_value=0)
    _patch_old_graphs(mocker, {1: "telegram-on-message"})

    with caplog.at_level(logging.WARNING, logger="backend.api.features.library.db"):
        count = await migrate_webhook_presets_to_new_version(
            user_id="user-789", new_graph=graph
        )

    assert count == 0
    assert any(
        "Not migrating preset #bad" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Call-site tests: verify the migration helper is invoked from every publish
# pathway when the new graph version carries a webhook input node, and is
# skipped when it does not.
# ---------------------------------------------------------------------------


def _make_graph_mock(*, has_webhook: bool, version: int = 2, is_active: bool = True):
    """Return a stand-in for a GraphModel that satisfies the call-site code."""
    graph = MagicMock()
    graph.id = "graph-xyz"
    graph.version = version
    graph.is_active = is_active
    graph.webhook_input_node = object() if has_webhook else None
    # reassign_ids / validate_graph are sync methods on the real model; keep
    # the MagicMock default (returns MagicMock) so they don't trigger async
    # warnings.
    return graph


@pytest.mark.asyncio
async def test_update_graph_in_library_migrates_when_webhook_node_present(
    mocker,
):
    """``update_graph_in_library`` should call the migration helper."""
    from backend.api.features.library import db as library_db

    new_graph = _make_graph_mock(has_webhook=True)
    incoming = AsyncMock()
    incoming.id = new_graph.id

    mocker.patch.object(library_db.graph_db, "get_graph_all_versions", return_value=[])
    mocker.patch.object(library_db.graph_db, "make_graph_model", return_value=new_graph)
    mocker.patch.object(library_db.graph_db, "create_graph", return_value=new_graph)
    mocker.patch.object(
        library_db, "get_library_agent_by_graph_id", return_value=AsyncMock()
    )
    mocker.patch.object(
        library_db,
        "update_library_agent_version_and_settings",
        return_value=AsyncMock(),
    )
    mocker.patch.object(
        library_db, "before_graph_activate", side_effect=lambda g, user_id: g
    )
    mocker.patch.object(library_db, "on_graph_deactivate", return_value=None)
    mocker.patch.object(library_db.graph_db, "set_graph_active_version")
    migrate_mock = mocker.patch.object(
        library_db,
        "migrate_webhook_presets_to_new_version",
        return_value=1,
    )

    await library_db.update_graph_in_library(graph=incoming, user_id="user-1")

    migrate_mock.assert_awaited_once_with(
        user_id="user-1",
        new_graph=new_graph,
    )


@pytest.mark.asyncio
async def test_update_graph_in_library_skips_when_no_webhook_node(mocker):
    """No migration call when the new graph has no webhook input node."""
    from backend.api.features.library import db as library_db

    new_graph = _make_graph_mock(has_webhook=False)
    incoming = AsyncMock()
    incoming.id = new_graph.id

    mocker.patch.object(library_db.graph_db, "get_graph_all_versions", return_value=[])
    mocker.patch.object(library_db.graph_db, "make_graph_model", return_value=new_graph)
    mocker.patch.object(library_db.graph_db, "create_graph", return_value=new_graph)
    mocker.patch.object(
        library_db, "get_library_agent_by_graph_id", return_value=AsyncMock()
    )
    mocker.patch.object(
        library_db,
        "update_library_agent_version_and_settings",
        return_value=AsyncMock(),
    )
    mocker.patch.object(
        library_db, "before_graph_activate", side_effect=lambda g, user_id: g
    )
    mocker.patch.object(library_db, "on_graph_deactivate", return_value=None)
    mocker.patch.object(library_db.graph_db, "set_graph_active_version")
    migrate_mock = mocker.patch.object(
        library_db,
        "migrate_webhook_presets_to_new_version",
        return_value=0,
    )

    await library_db.update_graph_in_library(graph=incoming, user_id="user-1")

    migrate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_v1_update_graph_migrates_when_webhook_node_present(mocker):
    """The PUT /graphs/{id} route triggers migration on the new active version."""
    from backend.api.features import v1

    new_graph = _make_graph_mock(has_webhook=True, version=3)
    incoming = AsyncMock()
    incoming.id = new_graph.id

    existing_version = MagicMock()
    existing_version.version = new_graph.version - 1
    existing_version.is_active = True
    mocker.patch.object(
        v1.graph_db,
        "get_graph_all_versions",
        return_value=[existing_version],
    )
    mocker.patch.object(v1.graph_db, "make_graph_model", return_value=new_graph)
    mocker.patch.object(v1.graph_db, "create_graph", return_value=new_graph)
    mocker.patch.object(v1.graph_db, "set_graph_active_version")
    mocker.patch.object(v1.graph_db, "get_graph", return_value=new_graph)
    mocker.patch.object(
        v1.library_db,
        "update_library_agent_version_and_settings",
        return_value=AsyncMock(),
    )
    migrate_mock = mocker.patch.object(
        v1.library_db,
        "migrate_webhook_presets_to_new_version",
        return_value=1,
    )
    mocker.patch.object(v1, "before_graph_activate", side_effect=lambda g, user_id: g)
    mocker.patch.object(v1, "on_graph_deactivate", return_value=None)

    await v1.update_graph(graph_id=new_graph.id, graph=incoming, user_id="user-1")

    migrate_mock.assert_awaited_once_with(
        user_id="user-1",
        new_graph=new_graph,
    )


@pytest.mark.asyncio
async def test_v1_update_graph_skips_when_no_webhook_node(mocker):
    """No migration call from PUT /graphs/{id} when graph has no webhook node."""
    from backend.api.features import v1

    new_graph = _make_graph_mock(has_webhook=False, version=3)
    incoming = AsyncMock()
    incoming.id = new_graph.id

    existing_version = MagicMock()
    existing_version.version = new_graph.version - 1
    existing_version.is_active = True
    mocker.patch.object(
        v1.graph_db,
        "get_graph_all_versions",
        return_value=[existing_version],
    )
    mocker.patch.object(v1.graph_db, "make_graph_model", return_value=new_graph)
    mocker.patch.object(v1.graph_db, "create_graph", return_value=new_graph)
    mocker.patch.object(v1.graph_db, "set_graph_active_version")
    mocker.patch.object(v1.graph_db, "get_graph", return_value=new_graph)
    mocker.patch.object(
        v1.library_db,
        "update_library_agent_version_and_settings",
        return_value=AsyncMock(),
    )
    migrate_mock = mocker.patch.object(
        v1.library_db,
        "migrate_webhook_presets_to_new_version",
        return_value=0,
    )
    mocker.patch.object(v1, "before_graph_activate", side_effect=lambda g, user_id: g)
    mocker.patch.object(v1, "on_graph_deactivate", return_value=None)

    await v1.update_graph(graph_id=new_graph.id, graph=incoming, user_id="user-1")

    migrate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_v1_set_graph_active_version_migrates_when_webhook_node_present(
    mocker,
):
    """PUT /graphs/{id}/versions/active triggers migration on the activated version."""
    from backend.api.features import v1

    target_graph = _make_graph_mock(has_webhook=True, version=4)

    mocker.patch.object(
        v1.graph_db,
        "get_graph",
        side_effect=[target_graph, target_graph],
    )
    mocker.patch.object(v1.graph_db, "set_graph_active_version")
    mocker.patch.object(
        v1.library_db,
        "update_library_agent_version_and_settings",
        return_value=AsyncMock(),
    )
    migrate_mock = mocker.patch.object(
        v1.library_db,
        "migrate_webhook_presets_to_new_version",
        return_value=2,
    )
    mocker.patch.object(v1, "before_graph_activate", side_effect=lambda g, user_id: g)
    mocker.patch.object(v1, "on_graph_deactivate", return_value=None)

    body = v1.SetGraphActiveVersion(active_graph_version=target_graph.version)

    await v1.set_graph_active_version(
        graph_id=target_graph.id, request_body=body, user_id="user-1"
    )

    migrate_mock.assert_awaited_once_with(
        user_id="user-1",
        new_graph=target_graph,
    )


@pytest.mark.asyncio
async def test_v1_set_graph_active_version_skips_when_no_webhook_node(mocker):
    """PUT /graphs/{id}/versions/active skips migration when no webhook node."""
    from backend.api.features import v1

    target_graph = _make_graph_mock(has_webhook=False, version=4)

    mocker.patch.object(
        v1.graph_db,
        "get_graph",
        side_effect=[target_graph, target_graph],
    )
    mocker.patch.object(v1.graph_db, "set_graph_active_version")
    mocker.patch.object(
        v1.library_db,
        "update_library_agent_version_and_settings",
        return_value=AsyncMock(),
    )
    migrate_mock = mocker.patch.object(
        v1.library_db,
        "migrate_webhook_presets_to_new_version",
        return_value=0,
    )
    mocker.patch.object(v1, "before_graph_activate", side_effect=lambda g, user_id: g)
    mocker.patch.object(v1, "on_graph_deactivate", return_value=None)

    body = v1.SetGraphActiveVersion(active_graph_version=target_graph.version)

    await v1.set_graph_active_version(
        graph_id=target_graph.id, request_body=body, user_id="user-1"
    )

    migrate_mock.assert_not_awaited()
