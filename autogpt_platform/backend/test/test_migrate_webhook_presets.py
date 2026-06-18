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


@pytest.mark.asyncio
async def test_migrate_updates_matching_presets(mock_prisma):
    mock_prisma.update_many = AsyncMock(return_value=3)

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", graph_id="graph-abc", new_version=5
    )

    assert count == 3
    mock_prisma.update_many.assert_called_once_with(
        where={
            "userId": "user-123",
            "agentGraphId": "graph-abc",
            "agentGraphVersion": {"lt": 5},
            "webhookId": {"not": None},
            "isDeleted": False,
        },
        data={"agentGraphVersion": 5},
    )


@pytest.mark.asyncio
async def test_migrate_returns_zero_when_no_matches(mock_prisma):
    mock_prisma.update_many = AsyncMock(return_value=0)

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", graph_id="graph-abc", new_version=1
    )

    assert count == 0


@pytest.mark.asyncio
async def test_migrate_filters_correctly(mock_prisma):
    mock_prisma.update_many = AsyncMock(return_value=1)

    await migrate_webhook_presets_to_new_version(
        user_id="user-456", graph_id="graph-xyz", new_version=10
    )

    where = mock_prisma.update_many.call_args.kwargs["where"]
    assert where["webhookId"] == {"not": None}
    assert where["isDeleted"] is False
    assert where["userId"] == "user-456"
    assert where["agentGraphId"] == "graph-xyz"
    # Only strictly older versions should be migrated (not equal, not newer).
    assert where["agentGraphVersion"] == {"lt": 10}


@pytest.mark.asyncio
async def test_migrate_logs_when_presets_are_migrated(mock_prisma, caplog):
    """Exercise the ``count > 0`` log branch."""
    mock_prisma.update_many = AsyncMock(return_value=2)

    with caplog.at_level(logging.INFO, logger="backend.api.features.library.db"):
        count = await migrate_webhook_presets_to_new_version(
            user_id="user-789", graph_id="graph-log", new_version=4
        )

    assert count == 2
    # The function logs an INFO message when at least one preset is migrated.
    assert any(
        "Migrated 2 webhook preset(s)" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_migrate_does_not_log_when_nothing_changes(mock_prisma, caplog):
    """The ``count > 0`` log branch is skipped when no presets matched."""
    mock_prisma.update_many = AsyncMock(return_value=0)

    with caplog.at_level(logging.INFO, logger="backend.api.features.library.db"):
        count = await migrate_webhook_presets_to_new_version(
            user_id="user-789", graph_id="graph-nolog", new_version=4
        )

    assert count == 0
    assert not any("Migrated" in record.message for record in caplog.records)


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
        graph_id=new_graph.id,
        new_version=new_graph.version,
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
        graph_id=new_graph.id,
        new_version=new_graph.version,
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
        graph_id=target_graph.id,
        new_version=target_graph.version,
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
