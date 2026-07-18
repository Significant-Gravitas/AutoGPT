"""Real-DB tests for admin write operations (models, migrations, routes)."""

from __future__ import annotations

import prisma.models
import pytest
from prisma.enums import LlmCatalogSource

from backend.data.llm_registry import db_routes, db_write
from backend.data.llm_registry.db_routes import UnknownRouteModelError

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(autouse=True)
async def clean_registry_tables(server):
    """Empty all registry tables before each test (FK-safe order)."""
    await prisma.models.LlmModelRoute.prisma().delete_many()
    await prisma.models.LlmModelMigration.prisma().delete_many()
    await prisma.models.LlmModelCost.prisma().delete_many()
    await prisma.models.LlmModel.prisma().update_many(
        where={}, data={"fallbackModelSlug": None}
    )
    await prisma.models.LlmModel.prisma().delete_many()
    await prisma.models.LlmProvider.prisma().delete_many()
    await prisma.models.LlmModelCreator.prisma().delete_many()
    await prisma.models.LlmCatalogState.prisma().delete_many()


async def _provider() -> prisma.models.LlmProvider:
    return await prisma.models.LlmProvider.prisma().create(
        data={"name": "openai", "displayName": "OpenAI"}
    )


async def _model(
    provider_id: str, slug: str = "openai/gpt-a", **overrides
) -> prisma.models.LlmModel:
    data = {
        "slug": slug,
        "displayName": slug,
        "Provider": {"connect": {"id": provider_id}},
        "contextWindow": 128000,
        "priceTier": 2,
    }
    data.update(overrides)
    return await prisma.models.LlmModel.prisma().create(data=data)


async def test_create_model_claims_local_and_maps_new_columns():
    provider = await _provider()
    await _model(provider.id, slug="openai/fallback-target")

    model = await db_write.create_model(
        slug="openai/gpt-new",
        display_name="GPT New",
        provider_id=provider.id,
        context_window=200000,
        price_tier=3,
        visibility="HIDDEN",
        min_subscription_tier="PRO",
        fallback_model_slug="openai/fallback-target",
    )

    assert model.source == LlmCatalogSource.LOCAL
    assert str(model.visibility) == "HIDDEN"
    assert str(model.minSubscriptionTier) == "PRO"
    assert model.fallbackModelSlug == "openai/fallback-target"


async def test_create_model_rejects_unknown_fallback():
    provider = await _provider()

    with pytest.raises(ValueError, match="does not exist"):
        await db_write.create_model(
            slug="openai/gpt-new",
            display_name="GPT New",
            provider_id=provider.id,
            context_window=200000,
            price_tier=1,
            fallback_model_slug="openai/no-such-model",
        )


async def test_update_model_claims_local_and_enforces_single_recommended():
    provider = await _provider()
    a = await _model(provider.id, slug="openai/gpt-a", isRecommended=True)
    b = await _model(provider.id, slug="openai/gpt-b")

    updated = await db_write.update_model(b.id, is_recommended=True)

    assert updated.isRecommended is True
    assert updated.source == LlmCatalogSource.LOCAL
    a_after = await prisma.models.LlmModel.prisma().find_unique(where={"id": a.id})
    assert a_after is not None and a_after.isRecommended is False


async def test_get_model_usage_runs_schema_prefixed_query():
    assert await db_write.get_model_usage("openai/gpt-a") == {
        "model_slug": "openai/gpt-a",
        "node_count": 0,
    }


async def test_toggle_disable_without_migration_claims_local():
    provider = await _provider()
    model = await _model(provider.id)

    result = await db_write.toggle_model_with_migration(model.id, is_enabled=False)

    assert result["nodes_migrated"] == 0
    after = await prisma.models.LlmModel.prisma().find_unique(where={"id": model.id})
    assert after is not None
    assert after.isEnabled is False
    assert after.source == LlmCatalogSource.LOCAL


async def test_toggle_with_disabled_replacement_rejected():
    provider = await _provider()
    model = await _model(provider.id)
    await _model(provider.id, slug="openai/gpt-dead", isEnabled=False)

    with pytest.raises(ValueError, match="disabled"):
        await db_write.toggle_model_with_migration(
            model.id, is_enabled=False, migrate_to_slug="openai/gpt-dead"
        )


async def test_toggle_with_migration_and_no_matching_nodes():
    provider = await _provider()
    model = await _model(provider.id)
    await _model(provider.id, slug="openai/gpt-replacement")

    result = await db_write.toggle_model_with_migration(
        model.id, is_enabled=False, migrate_to_slug="openai/gpt-replacement"
    )

    assert result["nodes_migrated"] == 0
    assert result["migration_id"] is None
    assert await prisma.models.LlmModelMigration.prisma().count() == 0


async def test_delete_unused_model():
    provider = await _provider()
    model = await _model(provider.id)

    result = await db_write.delete_model(model.id)

    assert result["deleted_model_slug"] == "openai/gpt-a"
    assert result["nodes_migrated"] == 0
    assert (
        await prisma.models.LlmModel.prisma().find_unique(where={"id": model.id})
        is None
    )


async def test_revert_unknown_migration_rejected():
    with pytest.raises(ValueError, match="not found"):
        await db_write.revert_migration("no-such-migration")


# --------------------------------------------------------------------------
# Routing cells
# --------------------------------------------------------------------------


async def test_set_route_upserts_and_lists():
    provider = await _provider()
    await _model(provider.id, supportsReasoning=True, supportsTools=True)

    row, warnings = await db_routes.set_route(
        "copilot", "thinking", "standard", "openai/gpt-a"
    )
    assert row is not None and row.modelSlug == "openai/gpt-a"
    assert warnings == []

    # Upsert same cell to a second model
    await _model(provider.id, slug="openai/gpt-b", supportsTools=True)
    row2, _ = await db_routes.set_route(
        "copilot", "thinking", "standard", "openai/gpt-b"
    )
    assert row2 is not None and row2.modelSlug == "openai/gpt-b"

    routes = await db_routes.list_routes()
    assert len(routes) == 1


async def test_set_route_unknown_model_raises_lookup():
    with pytest.raises(UnknownRouteModelError):
        await db_routes.set_route("copilot", "fast", "standard", "no/such-model")


async def test_set_route_disabled_model_rejected_hidden_allowed():
    provider = await _provider()
    await _model(provider.id, slug="openai/gpt-dead", isEnabled=False)
    await _model(
        provider.id, slug="openai/gpt-hidden", visibility="HIDDEN", supportsTools=True
    )

    with pytest.raises(ValueError, match="kill switch"):
        await db_routes.set_route("copilot", "fast", "standard", "openai/gpt-dead")

    row, warnings = await db_routes.set_route(
        "copilot", "fast", "standard", "openai/gpt-hidden"
    )
    assert row is not None and row.modelSlug == "openai/gpt-hidden"
    assert warnings == []


async def test_set_route_capability_warnings():
    provider = await _provider()
    await _model(provider.id, slug="openai/gpt-plain")  # no reasoning, no tools

    _, warnings = await db_routes.set_route(
        "copilot", "thinking", "standard", "openai/gpt-plain"
    )

    assert any("reasoning" in w for w in warnings)
    assert any("tool support" in w for w in warnings)


async def test_set_route_none_deletes_cell():
    provider = await _provider()
    await _model(provider.id, supportsTools=True)
    await db_routes.set_route("copilot", "fast", "standard", "openai/gpt-a")

    row, warnings = await db_routes.set_route("copilot", "fast", "standard", None)

    assert row is None and warnings == []
    assert await db_routes.list_routes() == []
