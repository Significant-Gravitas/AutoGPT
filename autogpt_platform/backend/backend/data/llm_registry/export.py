"""Export the registry DB as a CatalogPayload.

Used to regenerate the bundled ``catalog.json`` from a populated database
(``poetry run python -m backend.data.llm_registry.export > catalog.json``)
and by the admin catalog-export endpoint.

Only GA-visibility models are exported — the catalog is a distribution
artifact, and models mid-rollout (EMPLOYEES/ADMINS/HIDDEN) or admin-hidden
are not for distribution. Disabled GA models ARE exported so a remote
disable propagates to syncing installs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import prisma.models
from prisma.enums import LlmModelVisibility

import backend.data.db
from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogCreator,
    CatalogModel,
    CatalogPayload,
    CatalogProvider,
)


async def export_catalog() -> CatalogPayload:
    """Read the registry tables and build a canonical, sorted CatalogPayload."""
    providers = await prisma.models.LlmProvider.prisma().find_many()
    creators = await prisma.models.LlmModelCreator.prisma().find_many()
    models = await prisma.models.LlmModel.prisma().find_many(
        where={"visibility": LlmModelVisibility.GA},
        include={"Provider": True, "Creator": True},
    )
    return CatalogPayload(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        providers=sorted(
            (
                CatalogProvider(
                    name=p.name,
                    display_name=p.displayName,
                    description=p.description,
                    metadata=dict(p.metadata or {}),
                )
                for p in providers
            ),
            key=lambda p: p.name,
        ),
        creators=sorted(
            (
                CatalogCreator(
                    name=c.name,
                    display_name=c.displayName,
                    description=c.description,
                    website_url=c.websiteUrl,
                    logo_url=c.logoUrl,
                )
                for c in creators
            ),
            key=lambda c: c.name,
        ),
        models=sorted(
            (
                CatalogModel(
                    slug=m.slug,
                    display_name=m.displayName,
                    description=m.description,
                    provider=m.Provider.name if m.Provider else m.providerId,
                    creator=m.Creator.name if m.Creator else None,
                    kind=str(m.kind),
                    context_window=m.contextWindow,
                    max_output_tokens=m.maxOutputTokens,
                    price_tier=m.priceTier,
                    is_enabled=m.isEnabled,
                    is_recommended=m.isRecommended,
                    fallback_model_slug=m.fallbackModelSlug,
                    supports_tools=m.supportsTools,
                    supports_json_output=m.supportsJsonOutput,
                    supports_reasoning=m.supportsReasoning,
                    supports_parallel_tool_calls=m.supportsParallelToolCalls,
                    capabilities=dict(m.capabilities or {}),
                    metadata=dict(m.metadata or {}),
                )
                for m in models
            ),
            key=lambda m: m.slug,
        ),
    )


async def _main() -> None:
    await backend.data.db.connect()
    payload = await export_catalog()
    print(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
