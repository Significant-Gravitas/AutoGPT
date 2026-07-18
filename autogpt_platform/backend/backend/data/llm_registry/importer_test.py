"""Integration tests for the idempotent catalog importer (real DB).

Each test starts from empty registry tables. The merge-rule contract under
test: upsert-by-slug, LOCAL rows untouched, removals disable-not-delete,
content-hash fast-path, schema_version rejection.
"""

from __future__ import annotations

from datetime import datetime, timezone

import prisma.models
import pytest
from prisma.enums import LlmCatalogSource

from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogCreator,
    CatalogModel,
    CatalogPayload,
    CatalogProvider,
)
from backend.data.llm_registry.importer import (
    CatalogSchemaVersionError,
    import_bundled_catalog,
    import_catalog,
)

# Prisma's client is bound to the session-scoped server fixture's loop.
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


def _payload(models: list[CatalogModel]) -> CatalogPayload:
    return CatalogPayload(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        providers=[CatalogProvider(name="openai", display_name="OpenAI")],
        creators=[CatalogCreator(name="openai", display_name="OpenAI")],
        models=models,
    )


def _model(slug: str, **overrides) -> CatalogModel:
    kwargs = dict(
        slug=slug,
        display_name=slug.split("/")[-1],
        provider="openai",
        creator="openai",
        context_window=128000,
    )
    kwargs.update(overrides)
    return CatalogModel(**kwargs)


async def test_import_creates_everything():
    result = await import_catalog(
        _payload([_model("openai/gpt-a"), _model("openai/gpt-b")]),
        source=LlmCatalogSource.SEED,
    )

    assert result.unchanged is False
    assert result.providers_created == 1
    assert result.creators_created == 1
    assert result.models_created == 2

    rows = await prisma.models.LlmModel.prisma().find_many()
    assert {r.slug for r in rows} == {"openai/gpt-a", "openai/gpt-b"}
    assert all(r.source == LlmCatalogSource.SEED for r in rows)

    state = await prisma.models.LlmCatalogState.prisma().find_unique(
        where={"id": "singleton"}
    )
    assert state is not None
    assert state.contentHash == result.content_hash
    assert state.lastImportSource == LlmCatalogSource.SEED


async def test_reimport_same_payload_hits_hash_fast_path():
    payload = _payload([_model("openai/gpt-a")])
    first = await import_catalog(payload, source=LlmCatalogSource.SEED)
    second = await import_catalog(payload, source=LlmCatalogSource.SEED)

    assert first.unchanged is False
    assert second.unchanged is True
    assert second.content_hash == first.content_hash


async def test_local_rows_are_never_touched():
    await import_catalog(
        _payload([_model("openai/gpt-a"), _model("openai/gpt-b")]),
        source=LlmCatalogSource.SEED,
    )
    # Admin claims gpt-a and renames it.
    await prisma.models.LlmModel.prisma().update_many(
        where={"slug": "openai/gpt-a"},
        data={"source": LlmCatalogSource.LOCAL, "displayName": "Admin Edit"},
    )

    # New catalog renames both models.
    await import_catalog(
        _payload(
            [
                _model("openai/gpt-a", display_name="Catalog Rename A"),
                _model("openai/gpt-b", display_name="Catalog Rename B"),
            ]
        ),
        source=LlmCatalogSource.REMOTE,
    )

    a = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": "openai/gpt-a"}
    )
    b = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": "openai/gpt-b"}
    )
    assert a is not None and a.displayName == "Admin Edit"
    assert a.source == LlmCatalogSource.LOCAL
    assert b is not None and b.displayName == "Catalog Rename B"
    assert b.source == LlmCatalogSource.REMOTE


async def test_removed_models_are_disabled_not_deleted():
    await import_catalog(
        _payload([_model("openai/gpt-a"), _model("openai/gpt-old")]),
        source=LlmCatalogSource.SEED,
    )

    await import_catalog(
        _payload([_model("openai/gpt-a")]), source=LlmCatalogSource.SEED
    )

    old = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": "openai/gpt-old"}
    )
    assert old is not None, "removed model must not be deleted"
    assert old.isEnabled is False
    assert old.catalogRemovedAt is not None


async def test_removed_local_models_stay_enabled():
    await import_catalog(
        _payload([_model("openai/gpt-a"), _model("openai/gpt-mine")]),
        source=LlmCatalogSource.SEED,
    )
    await prisma.models.LlmModel.prisma().update_many(
        where={"slug": "openai/gpt-mine"}, data={"source": LlmCatalogSource.LOCAL}
    )

    await import_catalog(
        _payload([_model("openai/gpt-a")]), source=LlmCatalogSource.SEED
    )

    mine = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": "openai/gpt-mine"}
    )
    assert mine is not None
    assert mine.isEnabled is True
    assert mine.catalogRemovedAt is None


async def test_reappearing_model_is_reenabled():
    await import_catalog(
        _payload([_model("openai/gpt-a"), _model("openai/gpt-b")]),
        source=LlmCatalogSource.SEED,
    )
    await import_catalog(
        _payload([_model("openai/gpt-a")]), source=LlmCatalogSource.SEED
    )
    await import_catalog(
        _payload([_model("openai/gpt-a"), _model("openai/gpt-b")]),
        source=LlmCatalogSource.SEED,
    )

    b = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": "openai/gpt-b"}
    )
    assert b is not None
    assert b.isEnabled is True
    assert b.catalogRemovedAt is None


async def test_fallback_pointer_is_written_after_all_models_exist():
    await import_catalog(
        _payload(
            [
                # gpt-a's fallback references gpt-b which appears later in the
                # list — the two-pass write must handle forward references.
                _model("openai/gpt-a", fallback_model_slug="openai/gpt-b"),
                _model("openai/gpt-b"),
            ]
        ),
        source=LlmCatalogSource.SEED,
    )

    a = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": "openai/gpt-a"}
    )
    assert a is not None
    assert a.fallbackModelSlug == "openai/gpt-b"


async def test_schema_version_mismatch_is_rejected():
    payload = _payload([_model("openai/gpt-a")]).model_copy(
        update={"schema_version": CATALOG_SCHEMA_VERSION + 1}
    )
    with pytest.raises(CatalogSchemaVersionError):
        await import_catalog(payload, source=LlmCatalogSource.REMOTE)

    assert await prisma.models.LlmModel.prisma().count() == 0


async def test_bundled_catalog_imports_cleanly():
    result = await import_bundled_catalog()

    assert result.models_created > 50
    count = await prisma.models.LlmModel.prisma().count()
    assert count == result.models_created
