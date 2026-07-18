"""Idempotent catalog importer — runs on every startup.

Merge rules (the contract that makes catalog sync safe):
- Upsert by natural key (provider/creator ``name``, model ``slug``).
- Rows with ``source=LOCAL`` are admin-owned: never updated, never disabled.
  The guard lives INSIDE each UPDATE's WHERE clause so a concurrent admin
  write that claims a row LOCAL between our read and our write is not
  clobbered — do not "optimize" this into update-by-id.
- Models absent from the payload are disabled (``isEnabled=false`` +
  ``catalogRemovedAt``), never deleted. Providers/creators are never removed.
- ``LlmModelCost`` is never touched — cloud credit pricing is not catalog data.
- A sha256 over the canonical payload (minus ``generated_at``) short-circuits
  the steady-state boot to a single ``LlmCatalogState`` read.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib import resources

import prisma.models
from prisma.enums import LlmCatalogSource
from pydantic import BaseModel

from backend.data.db import transaction
from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogModel,
    CatalogPayload,
)
from backend.data.llm_registry.registry import refresh_runtime_caches

logger = logging.getLogger(__name__)

_NOT_LOCAL = {"not": LlmCatalogSource.LOCAL}


class CatalogSchemaVersionError(ValueError):
    """Payload schema_version doesn't match this build's CATALOG_SCHEMA_VERSION."""


class ImportResult(BaseModel):
    unchanged: bool = False
    content_hash: str
    providers_created: int = 0
    creators_created: int = 0
    models_created: int = 0
    models_updated: int = 0
    models_disabled: int = 0


def _canonical_hash(payload: CatalogPayload) -> str:
    data = payload.model_dump(mode="json", exclude={"generated_at"})
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _model_fact_data(model: CatalogModel, provider_id: str, creator_id: str | None):
    return {
        "displayName": model.display_name,
        "description": model.description,
        "providerId": provider_id,
        "creatorId": creator_id,
        "kind": model.kind,
        "contextWindow": model.context_window,
        "maxOutputTokens": model.max_output_tokens,
        "priceTier": model.price_tier,
        "isEnabled": model.is_enabled,
        "isRecommended": model.is_recommended,
        "supportsTools": model.supports_tools,
        "supportsJsonOutput": model.supports_json_output,
        "supportsReasoning": model.supports_reasoning,
        "supportsParallelToolCalls": model.supports_parallel_tool_calls,
        "capabilities": prisma.Json(model.capabilities),
        "metadata": prisma.Json(model.metadata),
        "catalogRemovedAt": None,
    }


async def _sync_providers(tx, payload: CatalogPayload, source) -> tuple[dict, int]:
    existing = {
        p.name: p for p in await prisma.models.LlmProvider.prisma(tx).find_many()
    }
    created = 0
    ids: dict[str, str] = {}
    for p in payload.providers:
        row = existing.get(p.name)
        data = {
            "displayName": p.display_name,
            "description": p.description,
            "metadata": prisma.Json(p.metadata),
        }
        if row is None:
            rec = await prisma.models.LlmProvider.prisma(tx).create(
                data={"name": p.name, "source": source, **data}
            )
            ids[p.name] = rec.id
            created += 1
        else:
            ids[p.name] = row.id
            if row.source != LlmCatalogSource.LOCAL:
                await prisma.models.LlmProvider.prisma(tx).update_many(
                    where={"name": p.name, "source": _NOT_LOCAL},
                    data={**data, "source": source},
                )
    return ids, created


async def _sync_creators(tx, payload: CatalogPayload, source) -> tuple[dict, int]:
    existing = {
        c.name: c for c in await prisma.models.LlmModelCreator.prisma(tx).find_many()
    }
    created = 0
    ids: dict[str, str] = {}
    for c in payload.creators:
        row = existing.get(c.name)
        data = {
            "displayName": c.display_name,
            "description": c.description,
            "websiteUrl": c.website_url,
            "logoUrl": c.logo_url,
        }
        if row is None:
            rec = await prisma.models.LlmModelCreator.prisma(tx).create(
                data={"name": c.name, "source": source, **data}
            )
            ids[c.name] = rec.id
            created += 1
        else:
            ids[c.name] = row.id
            if row.source != LlmCatalogSource.LOCAL:
                await prisma.models.LlmModelCreator.prisma(tx).update_many(
                    where={"name": c.name, "source": _NOT_LOCAL},
                    data={**data, "source": source},
                )
    return ids, created


async def _sync_models(
    tx, payload: CatalogPayload, source, provider_ids: dict, creator_ids: dict
) -> tuple[int, int, int]:
    existing = {m.slug: m for m in await prisma.models.LlmModel.prisma(tx).find_many()}
    created = updated = disabled = 0
    for m in payload.models:
        provider_id = provider_ids.get(m.provider)
        if provider_id is None:
            logger.warning(
                "Catalog model %s references unknown provider %s — skipped",
                m.slug,
                m.provider,
            )
            continue
        creator_id = creator_ids.get(m.creator) if m.creator else None
        data = _model_fact_data(m, provider_id, creator_id)
        row = existing.get(m.slug)
        if row is None:
            await prisma.models.LlmModel.prisma(tx).create(
                data={"slug": m.slug, "source": source, **data}
            )
            created += 1
        elif row.source != LlmCatalogSource.LOCAL:
            await prisma.models.LlmModel.prisma(tx).update_many(
                where={"slug": m.slug, "source": _NOT_LOCAL},
                data={**data, "source": source},
            )
            updated += 1

    # Second pass: fallback pointers reference slugs, so every model must
    # exist before any pointer is written.
    for m in payload.models:
        if m.slug not in existing and provider_ids.get(m.provider) is None:
            continue
        await prisma.models.LlmModel.prisma(tx).update_many(
            where={"slug": m.slug, "source": _NOT_LOCAL},
            data={"fallbackModelSlug": m.fallback_model_slug},
        )

    # Removal pass: disable non-LOCAL models the catalog no longer contains.
    payload_slugs = [m.slug for m in payload.models]
    disabled = await prisma.models.LlmModel.prisma(tx).update_many(
        where={
            "slug": {"not_in": payload_slugs},
            "source": _NOT_LOCAL,
            "catalogRemovedAt": None,
        },
        data={"isEnabled": False, "catalogRemovedAt": datetime.now(timezone.utc)},
    )
    return created, updated, disabled


async def import_catalog(
    payload: CatalogPayload,
    *,
    source: LlmCatalogSource,
    source_url: str | None = None,
) -> ImportResult:
    """Import a validated catalog payload. Returns per-entity change counts."""
    if payload.schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogSchemaVersionError(
            f"catalog schema_version={payload.schema_version} unsupported "
            f"(this build speaks {CATALOG_SCHEMA_VERSION})"
        )

    content_hash = _canonical_hash(payload)
    state = await prisma.models.LlmCatalogState.prisma().find_unique(
        where={"id": "singleton"}
    )
    now = datetime.now(timezone.utc)
    state_update = {
        "schemaVersion": payload.schema_version,
        "contentHash": content_hash,
        "lastImportSource": source,
        "lastImportedAt": now,
        "sourceUrl": source_url,
    }
    if state and state.contentHash == content_hash:
        logger.info(
            "LLM catalog unchanged (hash %s) — skipping import", content_hash[:12]
        )
        return ImportResult(unchanged=True, content_hash=content_hash)

    async with transaction() as tx:
        provider_ids, providers_created = await _sync_providers(tx, payload, source)
        creator_ids, creators_created = await _sync_creators(tx, payload, source)
        models_created, models_updated, models_disabled = await _sync_models(
            tx, payload, source, provider_ids, creator_ids
        )
        await prisma.models.LlmCatalogState.prisma(tx).upsert(
            where={"id": "singleton"},
            data={
                "create": {"id": "singleton", **state_update},
                "update": state_update,
            },
        )

    await refresh_runtime_caches()

    result = ImportResult(
        content_hash=content_hash,
        providers_created=providers_created,
        creators_created=creators_created,
        models_created=models_created,
        models_updated=models_updated,
        models_disabled=models_disabled,
    )
    logger.info(
        "LLM catalog imported (source=%s hash=%s): +%d/+%d/+%d models/providers/"
        "creators created, %d models updated, %d disabled",
        source,
        content_hash[:12],
        result.models_created,
        result.providers_created,
        result.creators_created,
        result.models_updated,
        result.models_disabled,
    )
    return result


async def import_bundled_catalog() -> ImportResult:
    """Import the catalog.json shipped with this build (source=SEED)."""
    raw = (
        resources.files("backend.data.llm_registry")
        .joinpath("catalog.json")
        .read_text(encoding="utf-8")
    )
    payload = CatalogPayload.model_validate_json(raw)
    return await import_catalog(payload, source=LlmCatalogSource.SEED)
