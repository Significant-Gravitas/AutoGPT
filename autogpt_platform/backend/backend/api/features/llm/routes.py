"""Public LLM catalog endpoint.

Unauthenticated by design: the payload is public model facts only. Costs
(cloud billing config), non-GA-visibility models (in-rollout), and routing
cells (per-deployment config) are stripped — the response is what any
consumer may know: which models exist and what they can do. Feeds the
model picker (Phase B) and any external consumer.

Served entirely from the catalog's in-process L1 cache — no DB reads on the
request path. The path is allowlisted in ``CACHEABLE_PATHS``, so the security
middleware leaves our Cache-Control header alone; that also means error
responses MUST set ``no-store`` explicitly or a CDN could cache a 429.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import fastapi
import fastapi.responses

import backend.data.llm_registry as llm_registry
from backend.api.features.llm.rate_limit import check_catalog_rate_limit, get_client_ip
from backend.data.llm_registry import (
    CATALOG_SCHEMA_VERSION,
    CatalogCreator,
    CatalogModel,
    CatalogPayload,
    CatalogProvider,
)

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()

_CACHE_CONTROL_OK = "public, max-age=300, stale-while-revalidate=3600"
_CACHE_CONTROL_ERROR = "no-store, no-cache, must-revalidate, private"


def _build_catalog_payload() -> CatalogPayload:
    """Assemble the catalog from the L1 cache. GA-visibility models only."""
    models = [m for m in llm_registry.get_all_models() if m.visibility == "GA"]

    providers: dict[str, CatalogProvider] = {}
    creators: dict[str, CatalogCreator] = {}
    catalog_models: list[CatalogModel] = []
    included_slugs = {m.slug for m in models}
    for m in models:
        providers.setdefault(
            m.metadata.provider,
            CatalogProvider(
                name=m.metadata.provider,
                display_name=m.provider_display_name,
            ),
        )
        if m.creator is not None:
            creators.setdefault(
                m.creator.name,
                CatalogCreator(
                    name=m.creator.name,
                    display_name=m.creator.display_name,
                    description=m.creator.description,
                    website_url=m.creator.website_url,
                    logo_url=m.creator.logo_url,
                ),
            )
        catalog_models.append(
            CatalogModel(
                slug=m.slug,
                display_name=m.display_name,
                description=m.description,
                provider=m.metadata.provider,
                creator=m.creator.name if m.creator else None,
                kind=m.kind,
                context_window=m.metadata.context_window,
                max_output_tokens=m.metadata.max_output_tokens,
                price_tier=m.metadata.price_tier,
                is_enabled=m.is_enabled,
                is_recommended=m.is_recommended,
                min_subscription_tier=m.min_subscription_tier,
                # Nulled when the fallback references a model excluded from
                # this payload (non-GA) — no dangling refs for consumers.
                fallback_model_slug=(
                    m.fallback_model_slug
                    if m.fallback_model_slug in included_slugs
                    else None
                ),
                supports_tools=m.supports_tools,
                supports_json_output=m.supports_json_output,
                supports_reasoning=m.supports_reasoning,
                supports_parallel_tool_calls=m.supports_parallel_tool_calls,
                capabilities=m.capabilities,
                metadata=m.extra_metadata,
                # Deliberately NOT carried into the public payload:
                cost=None,  # cloud billing config
            )
        )

    return CatalogPayload(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        providers=sorted(providers.values(), key=lambda p: p.name),
        creators=sorted(creators.values(), key=lambda c: c.name),
        models=sorted(catalog_models, key=lambda m: m.slug),
        routing={},  # per-deployment config, not catalog facts
    )


@router.get("/catalog", response_model=CatalogPayload)
async def get_llm_catalog(request: fastapi.Request) -> fastapi.responses.Response:
    ip = get_client_ip(request)
    if not await check_catalog_rate_limit(ip):
        return fastapi.responses.JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={
                "Retry-After": "60",
                "Cache-Control": _CACHE_CONTROL_ERROR,
            },
        )

    payload = _build_catalog_payload()
    return fastapi.responses.JSONResponse(
        content=payload.model_dump(mode="json"),
        headers={"Cache-Control": _CACHE_CONTROL_OK},
    )
