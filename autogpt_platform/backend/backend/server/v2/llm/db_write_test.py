"""Tests for LLM registry DB write operations (db_write.py).

All functions under test are async; patch Prisma at the point of use.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from backend.server.v2.llm import db_write

_NOW = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    id: str = "prov-1",
    name: str = "openai",
    display_name: str = "OpenAI",
    models: list | None = None,
) -> Mock:
    p = Mock()
    p.id = id
    p.name = name
    p.displayName = display_name
    p.description = None
    p.defaultCredentialProvider = None
    p.defaultCredentialId = None
    p.defaultCredentialType = None
    p.metadata = {}
    p.createdAt = _NOW
    p.updatedAt = _NOW
    p.Models = models if models is not None else []
    return p


def _make_model(
    id: str = "model-1",
    slug: str = "gpt-4",
    display_name: str = "GPT-4",
    is_enabled: bool = True,
    is_recommended: bool = False,
) -> Mock:
    m = Mock()
    m.id = id
    m.slug = slug
    m.displayName = display_name
    m.description = None
    m.providerId = "prov-1"
    m.creatorId = None
    m.contextWindow = 128000
    m.maxOutputTokens = 4096
    m.priceTier = 2
    m.isEnabled = is_enabled
    m.isRecommended = is_recommended
    m.supportsTools = False
    m.supportsJsonOutput = False
    m.supportsReasoning = False
    m.supportsParallelToolCalls = False
    m.capabilities = {}
    m.metadata = {}
    m.createdAt = _NOW
    m.updatedAt = _NOW
    m.Costs = []
    m.Creator = None
    return m


def _make_migration(
    id: str = "mig-1",
    source_slug: str = "gpt-3",
    target_slug: str = "gpt-4",
    node_count: int = 3,
    migrated_node_ids: list | None = None,
    is_reverted: bool = False,
) -> Mock:
    mg = Mock()
    mg.id = id
    mg.sourceModelSlug = source_slug
    mg.targetModelSlug = target_slug
    mg.reason = "upgrade"
    mg.nodeCount = node_count
    mg.customCreditCost = None
    mg.isReverted = is_reverted
    mg.revertedAt = None
    mg.createdAt = _NOW
    mg.migratedNodeIds = migrated_node_ids if migrated_node_ids is not None else ["n1", "n2", "n3"]
    return mg


def _make_tx_ctx(tx: Mock) -> Mock:
    """Return an async context manager that yields tx."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=tx)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


# ---------------------------------------------------------------------------
# Provider operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_provider(mocker):
    """create_provider calls prisma.create and returns the new provider."""
    mock_provider = _make_provider()
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.create = AsyncMock(return_value=mock_provider)

    result = await db_write.create_provider(name="openai", display_name="OpenAI")

    assert result.name == "openai"
    assert result.id == "prov-1"


@pytest.mark.asyncio
async def test_update_provider(mocker):
    """update_provider fetches existing provider then calls update."""
    existing = _make_provider()
    updated = _make_provider(display_name="OpenAI v2")
    prisma_mock = mocker.patch("prisma.models.LlmProvider.prisma").return_value
    prisma_mock.find_unique = AsyncMock(return_value=existing)
    prisma_mock.update = AsyncMock(return_value=updated)

    result = await db_write.update_provider(
        provider_id="prov-1", display_name="OpenAI v2"
    )

    assert result.displayName == "OpenAI v2"
    prisma_mock.find_unique.assert_called_once_with(where={"id": "prov-1"})


@pytest.mark.asyncio
async def test_update_provider_not_found(mocker):
    """update_provider raises ValueError when the provider does not exist."""
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="not found"):
        await db_write.update_provider(provider_id="ghost")


@pytest.mark.asyncio
async def test_delete_provider_success(mocker):
    """delete_provider deletes the provider when it has no models."""
    existing = _make_provider(models=[])
    prisma_mock = mocker.patch("prisma.models.LlmProvider.prisma").return_value
    prisma_mock.find_unique = AsyncMock(return_value=existing)
    prisma_mock.delete = AsyncMock(return_value=existing)

    result = await db_write.delete_provider(provider_id="prov-1")

    assert result is True
    prisma_mock.delete.assert_called_once_with(where={"id": "prov-1"})


@pytest.mark.asyncio
async def test_delete_provider_has_models(mocker):
    """delete_provider raises ValueError when the provider has associated models."""
    model = _make_model()
    existing = _make_provider(models=[model])
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    with pytest.raises(ValueError, match="Cannot delete"):
        await db_write.delete_provider(provider_id="prov-1")


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_model(mocker):
    """create_model calls prisma.create with slug and display_name in data."""
    mock_model = _make_model()
    prisma_create = mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.create
    prisma_create = AsyncMock(return_value=mock_model)
    mocker.patch("prisma.models.LlmModel.prisma").return_value.create = prisma_create

    result = await db_write.create_model(
        slug="gpt-4",
        display_name="GPT-4",
        provider_id="prov-1",
        context_window=128000,
        price_tier=2,
    )

    assert result.slug == "gpt-4"
    call_kwargs = prisma_create.call_args
    data_arg = call_kwargs.kwargs.get("data") or call_kwargs.args[0]
    assert data_arg["slug"] == "gpt-4"
    assert data_arg["displayName"] == "GPT-4"


@pytest.mark.asyncio
async def test_update_model_recommended_clears_others(mocker):
    """update_model with is_recommended=True calls update_many to clear others first."""
    updated_model = _make_model(is_recommended=True)

    tx = AsyncMock()
    tx.llmmodel.update_many = AsyncMock()
    tx.llmmodel.update = AsyncMock(return_value=updated_model)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    result = await db_write.update_model(model_id="model-1", is_recommended=True)

    tx.llmmodel.update_many.assert_called_once_with(
        where={"id": {"not": "model-1"}},
        data={"isRecommended": False},
    )
    assert result.isRecommended is True


@pytest.mark.asyncio
async def test_update_model_recommended_false_no_clear(mocker):
    """update_model with is_recommended=False does NOT call update_many."""
    updated_model = _make_model(is_recommended=False)

    tx = AsyncMock()
    tx.llmmodel.update_many = AsyncMock()
    tx.llmmodel.update = AsyncMock(return_value=updated_model)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    await db_write.update_model(model_id="model-1", is_recommended=False)

    tx.llmmodel.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_not_found(mocker):
    """update_model raises ValueError when tx.llmmodel.update returns None."""
    tx = AsyncMock()
    tx.llmmodel.update_many = AsyncMock()
    tx.llmmodel.update = AsyncMock(return_value=None)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    with pytest.raises(ValueError, match="not found"):
        await db_write.update_model(model_id="ghost")


@pytest.mark.asyncio
async def test_get_model_usage(mocker):
    """get_model_usage parses query_raw result and returns node_count."""
    mock_client = Mock()
    mock_client.query_raw = AsyncMock(return_value=[{"count": "3"}])
    mocker.patch("prisma.get_client", return_value=mock_client)

    result = await db_write.get_model_usage("gpt-4")

    assert result["node_count"] == 3
    assert result["model_slug"] == "gpt-4"


# ---------------------------------------------------------------------------
# Delete model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_model_no_usage(mocker):
    """delete_model with no node usage deletes the model and returns nodes_migrated=0."""
    model = _make_model()
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    tx = AsyncMock()
    tx.query_raw = AsyncMock(return_value=[{"count": "0"}])
    tx.execute_raw = AsyncMock()
    tx.llmmodel.delete = AsyncMock()
    tx.llmmodel.find_unique = AsyncMock()

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    result = await db_write.delete_model(model_id="model-1")

    tx.llmmodel.delete.assert_called_once_with(where={"id": "model-1"})
    assert result["nodes_migrated"] == 0
    assert result["deleted_model_slug"] == "gpt-4"


@pytest.mark.asyncio
async def test_delete_model_with_replacement(mocker):
    """delete_model migrates nodes and deletes when replacement is valid and enabled."""
    model = _make_model(slug="gpt-3")
    replacement = _make_model(id="model-2", slug="gpt-4", is_enabled=True)

    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    tx = AsyncMock()
    tx.query_raw = AsyncMock(return_value=[{"count": "2"}])
    tx.llmmodel.find_unique = AsyncMock(return_value=replacement)
    tx.execute_raw = AsyncMock()
    tx.llmmodel.delete = AsyncMock()

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    result = await db_write.delete_model(
        model_id="model-1", replacement_model_slug="gpt-4"
    )

    tx.execute_raw.assert_called_once()
    tx.llmmodel.delete.assert_called_once()
    assert result["nodes_migrated"] == 2
    assert result["replacement_model_slug"] == "gpt-4"


@pytest.mark.asyncio
async def test_delete_model_usage_no_replacement(mocker):
    """delete_model raises ValueError when nodes use the model but no replacement given."""
    model = _make_model()
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    tx = AsyncMock()
    tx.query_raw = AsyncMock(return_value=[{"count": "5"}])

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    with pytest.raises(ValueError, match="provide a replacement"):
        await db_write.delete_model(model_id="model-1")


@pytest.mark.asyncio
async def test_delete_model_replacement_disabled(mocker):
    """delete_model raises ValueError when the replacement model is disabled."""
    model = _make_model(slug="gpt-3")
    disabled_replacement = _make_model(slug="gpt-4", is_enabled=False)

    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    tx = AsyncMock()
    tx.query_raw = AsyncMock(return_value=[{"count": "2"}])
    tx.llmmodel.find_unique = AsyncMock(return_value=disabled_replacement)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    with pytest.raises(ValueError, match="is disabled"):
        await db_write.delete_model(
            model_id="model-1", replacement_model_slug="gpt-4"
        )


@pytest.mark.asyncio
async def test_delete_model_not_found(mocker):
    """delete_model raises ValueError when the model does not exist."""
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="not found"):
        await db_write.delete_model(model_id="ghost")


# ---------------------------------------------------------------------------
# Toggle model with migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toggle_enable_no_migration(mocker):
    """toggle_model_with_migration enable without migrate_to_slug does simple update."""
    model = _make_model(is_enabled=False)
    prisma_mock = mocker.patch("prisma.models.LlmModel.prisma").return_value
    prisma_mock.find_unique = AsyncMock(return_value=model)
    prisma_mock.update = AsyncMock(return_value=model)

    result = await db_write.toggle_model_with_migration(
        model_id="model-1", is_enabled=True
    )

    prisma_mock.update.assert_called_once()
    assert result["nodes_migrated"] == 0
    assert result["migration_id"] is None


@pytest.mark.asyncio
async def test_toggle_disable_with_migration(mocker):
    """Disabling with migrate_to_slug creates migration record and returns nodes_migrated."""
    model = _make_model(slug="gpt-3", is_enabled=True)
    replacement = _make_model(id="model-2", slug="gpt-4", is_enabled=True)

    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    migration_record = Mock()
    migration_record.id = "mig-new"

    tx = AsyncMock()
    tx.llmmodel.find_unique = AsyncMock(return_value=replacement)
    tx.query_raw = AsyncMock(
        return_value=[{"id": "node-1"}, {"id": "node-2"}]
    )
    tx.execute_raw = AsyncMock()
    tx.llmmodel.update = AsyncMock()
    tx.llmmodelmigration.create = AsyncMock(return_value=migration_record)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    result = await db_write.toggle_model_with_migration(
        model_id="model-1",
        is_enabled=False,
        migrate_to_slug="gpt-4",
        migration_reason="upgrade",
    )

    assert result["nodes_migrated"] == 2
    assert result["migration_id"] == "mig-new"
    tx.llmmodelmigration.create.assert_called_once()


@pytest.mark.asyncio
async def test_toggle_disable_migration_target_not_found(mocker):
    """Disabling with nonexistent replacement raises ValueError."""
    model = _make_model(slug="gpt-3", is_enabled=True)
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    tx = AsyncMock()
    tx.llmmodel.find_unique = AsyncMock(return_value=None)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    with pytest.raises(ValueError, match="not found"):
        await db_write.toggle_model_with_migration(
            model_id="model-1", is_enabled=False, migrate_to_slug="ghost"
        )


@pytest.mark.asyncio
async def test_toggle_disable_migration_target_disabled(mocker):
    """Disabling with a disabled replacement raises ValueError."""
    model = _make_model(slug="gpt-3", is_enabled=True)
    disabled = _make_model(slug="gpt-4", is_enabled=False)

    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=model)

    tx = AsyncMock()
    tx.llmmodel.find_unique = AsyncMock(return_value=disabled)

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    with pytest.raises(ValueError, match="disabled"):
        await db_write.toggle_model_with_migration(
            model_id="model-1", is_enabled=False, migrate_to_slug="gpt-4"
        )


@pytest.mark.asyncio
async def test_toggle_disable_without_migration(mocker):
    """Disabling without migrate_to_slug just updates is_enabled, no nodes migrated."""
    model = _make_model(slug="gpt-3", is_enabled=True)
    prisma_mock = mocker.patch("prisma.models.LlmModel.prisma").return_value
    prisma_mock.find_unique = AsyncMock(return_value=model)
    prisma_mock.update = AsyncMock(return_value=model)

    result = await db_write.toggle_model_with_migration(
        model_id="model-1", is_enabled=False  # no migrate_to_slug
    )

    # Should do a simple update, no transaction, no migration record
    prisma_mock.update.assert_called_once_with(
        where={"id": "model-1"}, data={"isEnabled": False}
    )
    assert result["nodes_migrated"] == 0
    assert result["migration_id"] is None


# ---------------------------------------------------------------------------
# Migration operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_migrations_active_only(mocker):
    """list_migrations(include_reverted=False) passes where={isReverted: False}."""
    records = [_make_migration()]
    prisma_find_many = mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_many
    prisma_find_many = AsyncMock(return_value=records)
    mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_many = prisma_find_many

    result = await db_write.list_migrations(include_reverted=False)

    call_kwargs = prisma_find_many.call_args
    where_arg = (call_kwargs.kwargs.get("where") or
                 (call_kwargs.args[0] if call_kwargs.args else None))
    assert where_arg == {"isReverted": False}
    assert len(result) == 1


@pytest.mark.asyncio
async def test_list_migrations_include_reverted(mocker):
    """list_migrations(include_reverted=True) passes where=None."""
    records = [_make_migration(), _make_migration(id="mig-2", is_reverted=True)]
    prisma_find_many = AsyncMock(return_value=records)
    mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_many = prisma_find_many

    result = await db_write.list_migrations(include_reverted=True)

    call_kwargs = prisma_find_many.call_args
    where_arg = call_kwargs.kwargs.get("where")
    assert where_arg is None
    assert len(result) == 2


@pytest.mark.asyncio
async def test_revert_migration_success(mocker):
    """revert_migration re-enables source model, updates nodes, marks migration reverted."""
    migration = _make_migration(migrated_node_ids=["n1", "n2"])
    source_model = _make_model(is_enabled=False)

    prisma_migration_mock = mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value
    prisma_migration_mock.find_unique = AsyncMock(return_value=migration)

    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=source_model)

    tx = AsyncMock()
    tx.llmmodel.update = AsyncMock()
    tx.execute_raw = AsyncMock(return_value=2)
    tx.llmmodelmigration.update = AsyncMock()

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    result = await db_write.revert_migration(
        migration_id="mig-1", re_enable_source_model=True
    )

    assert result["nodes_reverted"] == 2
    assert result["source_model_re_enabled"] is True
    tx.llmmodel.update.assert_called_once_with(
        where={"id": source_model.id},
        data={"isEnabled": True},
    )
    tx.llmmodelmigration.update.assert_called_once()


@pytest.mark.asyncio
async def test_revert_migration_already_reverted(mocker):
    """revert_migration raises ValueError when migration is already reverted."""
    migration = _make_migration(is_reverted=True)
    mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_unique = AsyncMock(return_value=migration)

    with pytest.raises(ValueError, match="already been reverted"):
        await db_write.revert_migration(migration_id="mig-1")


@pytest.mark.asyncio
async def test_revert_migration_not_found(mocker):
    """revert_migration raises ValueError when migration does not exist."""
    mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="not found"):
        await db_write.revert_migration(migration_id="ghost")


@pytest.mark.asyncio
async def test_revert_migration_no_re_enable(mocker):
    """revert_migration with re_enable_source_model=False does not re-enable source."""
    migration = _make_migration(migrated_node_ids=["n1", "n2"])
    source_model = _make_model(is_enabled=False)

    mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_unique = AsyncMock(return_value=migration)
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=source_model)

    tx = AsyncMock()
    tx.llmmodel.update = AsyncMock()
    tx.execute_raw = AsyncMock(return_value=2)
    tx.llmmodelmigration.update = AsyncMock()

    mocker.patch(
        "backend.server.v2.llm.db_write.transaction",
        return_value=_make_tx_ctx(tx),
    )

    result = await db_write.revert_migration(
        migration_id="mig-1", re_enable_source_model=False
    )

    # Model should NOT have been re-enabled
    tx.llmmodel.update.assert_not_called()
    assert result["source_model_re_enabled"] is False
    assert result["nodes_reverted"] == 2


@pytest.mark.asyncio
async def test_revert_migration_source_model_gone(mocker):
    """revert_migration raises ValueError when the source model no longer exists."""
    migration = _make_migration()
    mocker.patch(
        "prisma.models.LlmModelMigration.prisma"
    ).return_value.find_unique = AsyncMock(return_value=migration)
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="no longer exists"):
        await db_write.revert_migration(migration_id="mig-1")


# ---------------------------------------------------------------------------
# Cache refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_runtime_caches(mocker):
    """refresh_runtime_caches clears cache, refreshes registry, publishes notification."""
    mock_clear = mocker.patch("backend.data.llm_registry.clear_registry_cache")
    mock_refresh = mocker.patch(
        "backend.data.llm_registry.refresh_llm_registry",
        new=AsyncMock(),
    )
    mock_publish = mocker.patch(
        "backend.data.llm_registry.notifications.publish_registry_refresh_notification",
        new=AsyncMock(),
    )

    await db_write.refresh_runtime_caches()

    mock_clear.assert_called_once()
    mock_refresh.assert_called_once()
    mock_publish.assert_called_once()
