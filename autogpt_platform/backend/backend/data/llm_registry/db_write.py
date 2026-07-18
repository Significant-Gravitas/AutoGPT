"""Admin write operations for LLM registry models and migrations.

Every write stamps ``source=LOCAL`` — the admin-owns-this-row claim the
catalog importer respects (LOCAL rows are never updated or disabled by
catalog imports). Callers are responsible for invoking
``registry.refresh_runtime_caches()`` after a successful mutation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, LiteralString, cast

import prisma
import prisma.models
from prisma.enums import LlmCatalogSource

from backend.data.db import get_database_schema, query_raw_with_schema, transaction

logger = logging.getLogger(__name__)


def _schema_format(query_template: str) -> LiteralString:
    """Format a ``{schema_prefix}`` query for transaction-scoped raw SQL.

    ``query_raw_with_schema``/``execute_raw_with_schema`` can't run inside an
    existing transaction client for SELECTs, so FOR UPDATE reads format the
    template with the same prefix logic and go through ``tx.query_raw``.
    """
    schema = get_database_schema()
    schema_prefix = f'"{schema}".' if schema != "public" else ""
    # cast: prisma types raw queries as LiteralString; the template is a
    # module-level literal and the prefix derives from DATABASE_URL config.
    return cast(LiteralString, query_template.format(schema_prefix=schema_prefix))


def _node_model_value(slug: str) -> str:
    """Extract the model value stored in AgentNode.constantInput from a registry slug.

    Registry slugs are formatted as 'provider/model-name' (e.g. 'openai/gpt-4o').
    The LLM block stores only the model-name part (e.g. 'gpt-4o') in constantInput.
    """
    return slug.split("/", 1)[-1] if "/" in slug else slug


async def validate_fallback_slug(fallback_model_slug: str | None) -> None:
    """Fallback pointers must reference an existing model."""
    if fallback_model_slug is None:
        return
    row = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": fallback_model_slug}
    )
    if row is None:
        raise ValueError(
            f"Fallback model '{fallback_model_slug}' does not exist in the registry"
        )


def _build_model_data(
    slug: str,
    display_name: str,
    provider_id: str,
    context_window: int,
    price_tier: int,
    description: str | None = None,
    creator_id: str | None = None,
    max_output_tokens: int | None = None,
    is_enabled: bool = True,
    is_recommended: bool = False,
    kind: str = "CHAT",
    visibility: str = "GA",
    min_subscription_tier: str | None = None,
    fallback_model_slug: str | None = None,
    supports_tools: bool = False,
    supports_json_output: bool = False,
    supports_reasoning: bool = False,
    supports_parallel_tool_calls: bool = False,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build model data dict for Prisma operations. Always claims source=LOCAL."""
    data: dict[str, Any] = {
        "slug": slug,
        "displayName": display_name,
        "description": description,
        "Provider": {"connect": {"id": provider_id}},
        "contextWindow": context_window,
        "maxOutputTokens": max_output_tokens,
        "priceTier": price_tier,
        "isEnabled": is_enabled,
        "isRecommended": is_recommended,
        "kind": kind,
        "visibility": visibility,
        "minSubscriptionTier": min_subscription_tier,
        "supportsTools": supports_tools,
        "supportsJsonOutput": supports_json_output,
        "supportsReasoning": supports_reasoning,
        "supportsParallelToolCalls": supports_parallel_tool_calls,
        "capabilities": prisma.Json(capabilities or {}),
        "metadata": prisma.Json(metadata or {}),
        "source": LlmCatalogSource.LOCAL,
    }
    if creator_id:
        data["Creator"] = {"connect": {"id": creator_id}}
    if fallback_model_slug:
        # Relation-connect form: mixing the scalar FK with other relation
        # connects fails Prisma's checked-input union.
        data["FallbackModel"] = {"connect": {"slug": fallback_model_slug}}
    # Prisma create input rejects explicit None for optional fields —
    # omitting them yields the same NULL/default.
    return {k: v for k, v in data.items() if v is not None}


async def create_model(**kwargs: Any) -> prisma.models.LlmModel:
    """Create a new LLM model (admin-owned: source=LOCAL)."""
    await validate_fallback_slug(kwargs.get("fallback_model_slug"))
    data = _build_model_data(**kwargs)
    model = await prisma.models.LlmModel.prisma().create(
        data=data,
        include={"Costs": True, "Creator": True, "Provider": True},
    )
    if not model:
        raise ValueError("Failed to create model")
    return model


_UPDATE_FIELD_MAP = {
    "display_name": "displayName",
    "description": "description",
    "context_window": "contextWindow",
    "max_output_tokens": "maxOutputTokens",
    "price_tier": "priceTier",
    "is_enabled": "isEnabled",
    "is_recommended": "isRecommended",
    "kind": "kind",
    "visibility": "visibility",
    "min_subscription_tier": "minSubscriptionTier",
    "fallback_model_slug": "fallbackModelSlug",
    "supports_tools": "supportsTools",
    "supports_json_output": "supportsJsonOutput",
    "supports_reasoning": "supportsReasoning",
    "supports_parallel_tool_calls": "supportsParallelToolCalls",
}


async def update_model(model_id: str, **fields: Any) -> prisma.models.LlmModel:
    """Update an existing LLM model; the touch claims the row (source=LOCAL).

    When is_recommended=True, clears the flag on all other models first so
    only one model can be recommended at a time.
    """
    if "fallback_model_slug" in fields:
        await validate_fallback_slug(fields["fallback_model_slug"])

    data: dict[str, Any] = {"source": LlmCatalogSource.LOCAL}
    for key, column in _UPDATE_FIELD_MAP.items():
        if fields.get(key) is not None:
            data[column] = fields[key]
    if fields.get("capabilities") is not None:
        data["capabilities"] = prisma.Json(fields["capabilities"])
    if fields.get("metadata") is not None:
        data["metadata"] = prisma.Json(fields["metadata"])
    if fields.get("creator_id") is not None:
        data["creatorId"] = fields["creator_id"] or None

    async with transaction() as tx:
        # Enforce single recommended model: unset all others first.
        if fields.get("is_recommended") is True:
            await tx.llmmodel.update_many(
                where={"id": {"not": model_id}},
                data={"isRecommended": False},
            )
        model = await tx.llmmodel.update(
            where={"id": model_id},
            data=data,
            include={"Costs": True, "Creator": True, "Provider": True},
        )

    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")
    return model


async def get_model_usage(slug: str) -> dict[str, Any]:
    """Get usage count for a model — how many AgentNodes reference it."""
    model_value = _node_model_value(slug)
    count_result = await query_raw_with_schema(
        """
        SELECT COUNT(*) as count
        FROM {schema_prefix}"AgentNode"
        WHERE "constantInput"::jsonb->>'model' = $1
        """,
        model_value,
    )
    node_count = int(count_result[0]["count"]) if count_result else 0
    return {"model_slug": slug, "node_count": node_count}


async def _migrate_nodes_in_tx(tx, source_value: str, target_value: str) -> list[str]:
    """Lock + rewrite AgentNode model references. Returns migrated node ids."""
    node_ids_result = await tx.query_raw(
        _schema_format(
            """
            SELECT id
            FROM {schema_prefix}"AgentNode"
            WHERE "constantInput"::jsonb->>'model' = $1
            FOR UPDATE
            """
        ),
        source_value,
    )
    migrated_node_ids = [row["id"] for row in node_ids_result or []]
    if migrated_node_ids:
        await tx.execute_raw(
            _schema_format(
                """
                UPDATE {schema_prefix}"AgentNode"
                SET "constantInput" = JSONB_SET(
                    "constantInput"::jsonb,
                    '{{model}}',
                    to_jsonb($1::text)
                )
                WHERE id::text IN (
                    SELECT jsonb_array_elements_text($2::jsonb)
                )
                """
            ),
            target_value,
            json.dumps(migrated_node_ids),
        )
    return migrated_node_ids


async def _validate_replacement_in_tx(tx, slug: str) -> prisma.models.LlmModel:
    replacement = await tx.llmmodel.find_unique(where={"slug": slug})
    if not replacement:
        raise ValueError(f"Replacement model '{slug}' not found")
    if not replacement.isEnabled:
        raise ValueError(
            f"Replacement model '{slug}' is disabled. "
            f"Please enable it before using it as a replacement."
        )
    return replacement


async def toggle_model_with_migration(
    model_id: str,
    is_enabled: bool,
    migrate_to_slug: str | None = None,
    migration_reason: str | None = None,
    custom_credit_cost: int | None = None,
) -> dict[str, Any]:
    """Toggle a model's enabled status, optionally migrating workflows when disabling."""
    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}, include={"Costs": True}
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    toggle_data: dict[str, Any] = {
        "isEnabled": is_enabled,
        "source": LlmCatalogSource.LOCAL,
    }
    nodes_migrated = 0
    migration_id: str | None = None

    if not is_enabled and migrate_to_slug:
        async with transaction() as tx:
            await _validate_replacement_in_tx(tx, migrate_to_slug)
            migrated_node_ids = await _migrate_nodes_in_tx(
                tx,
                _node_model_value(model.slug),
                _node_model_value(migrate_to_slug),
            )
            nodes_migrated = len(migrated_node_ids)
            await tx.llmmodel.update(where={"id": model_id}, data=toggle_data)
            if nodes_migrated > 0:
                migration_record = await tx.llmmodelmigration.create(
                    data={
                        "sourceModelSlug": model.slug,
                        "targetModelSlug": migrate_to_slug,
                        "reason": migration_reason,
                        "migratedNodeIds": json.dumps(migrated_node_ids),
                        "nodeCount": nodes_migrated,
                        "customCreditCost": custom_credit_cost,
                    }
                )
                migration_id = migration_record.id
    else:
        await prisma.models.LlmModel.prisma().update(
            where={"id": model_id}, data=toggle_data
        )

    return {
        "nodes_migrated": nodes_migrated,
        "migrated_to_slug": migrate_to_slug if nodes_migrated > 0 else None,
        "migration_id": migration_id,
    }


async def delete_model(
    model_id: str, replacement_model_slug: str | None = None
) -> dict[str, Any]:
    """Delete an LLM model, optionally migrating affected AgentNodes first.

    If workflows are using this model and no replacement is given, raises
    ValueError. If replacement is given, atomically migrates all affected
    nodes then deletes.
    """
    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}, include={"Costs": True}
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    async with transaction() as tx:
        count_result = await tx.query_raw(
            _schema_format(
                """
                SELECT COUNT(*) as count
                FROM {schema_prefix}"AgentNode"
                WHERE "constantInput"::jsonb->>'model' = $1
                """
            ),
            _node_model_value(model.slug),
        )
        nodes_to_migrate = int(count_result[0]["count"]) if count_result else 0

        if nodes_to_migrate > 0:
            if not replacement_model_slug:
                raise ValueError(
                    f"Cannot delete model '{model.slug}': {nodes_to_migrate} "
                    f"workflow node(s) are using it. Please provide a "
                    f"replacement_model_slug to migrate them."
                )
            await _validate_replacement_in_tx(tx, replacement_model_slug)
            await _migrate_nodes_in_tx(
                tx,
                _node_model_value(model.slug),
                _node_model_value(replacement_model_slug),
            )
        await tx.llmmodel.delete(where={"id": model_id})

    return {
        "deleted_model_slug": model.slug,
        "deleted_model_display_name": model.displayName,
        "replacement_model_slug": replacement_model_slug,
        "nodes_migrated": nodes_to_migrate,
    }


async def list_migrations(include_reverted: bool = False) -> list[dict[str, Any]]:
    """List model migrations."""
    where: Any = None if include_reverted else {"isReverted": False}
    records = await prisma.models.LlmModelMigration.prisma().find_many(
        where=where,
        order={"createdAt": "desc"},
    )
    return [
        {
            "id": r.id,
            "source_model_slug": r.sourceModelSlug,
            "target_model_slug": r.targetModelSlug,
            "reason": r.reason,
            "node_count": r.nodeCount,
            "custom_credit_cost": r.customCreditCost,
            "is_reverted": r.isReverted,
            "reverted_at": r.revertedAt.isoformat() if r.revertedAt else None,
            "created_at": r.createdAt.isoformat(),
        }
        for r in records
    ]


async def revert_migration(
    migration_id: str,
    re_enable_source_model: bool = True,
) -> dict[str, Any]:
    """Revert a model migration, restoring affected nodes to their original model."""
    migration = await prisma.models.LlmModelMigration.prisma().find_unique(
        where={"id": migration_id}
    )
    if not migration:
        raise ValueError(f"Migration with id '{migration_id}' not found")
    if migration.isReverted:
        raise ValueError(f"Migration '{migration_id}' has already been reverted")

    source_model = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": migration.sourceModelSlug}
    )
    if not source_model:
        raise ValueError(
            f"Source model '{migration.sourceModelSlug}' no longer exists."
        )

    migrated_node_ids: list[str] = (
        migration.migratedNodeIds
        if isinstance(migration.migratedNodeIds, list)
        else json.loads(str(migration.migratedNodeIds))
    )
    if not migrated_node_ids:
        raise ValueError("No nodes to revert in this migration")

    source_model_re_enabled = False

    async with transaction() as tx:
        if not source_model.isEnabled and re_enable_source_model:
            await tx.llmmodel.update(
                where={"id": source_model.id},
                data={"isEnabled": True, "source": LlmCatalogSource.LOCAL},
            )
            source_model_re_enabled = True

        result = await tx.execute_raw(
            _schema_format(
                """
                UPDATE {schema_prefix}"AgentNode"
                SET "constantInput" = JSONB_SET(
                    "constantInput"::jsonb,
                    '{{model}}',
                    to_jsonb($1::text)
                )
                WHERE id::text IN (
                    SELECT jsonb_array_elements_text($2::jsonb)
                )
                AND "constantInput"::jsonb->>'model' = $3
                """
            ),
            _node_model_value(migration.sourceModelSlug),
            json.dumps(migrated_node_ids),
            _node_model_value(migration.targetModelSlug),
        )
        nodes_reverted = result if isinstance(result, int) else 0

        await tx.llmmodelmigration.update(
            where={"id": migration_id},
            data={"isReverted": True, "revertedAt": datetime.now(timezone.utc)},
        )

    return {
        "migration_id": migration_id,
        "source_model_slug": migration.sourceModelSlug,
        "target_model_slug": migration.targetModelSlug,
        "nodes_reverted": nodes_reverted,
        "nodes_already_changed": len(migrated_node_ids) - nodes_reverted,
        "source_model_re_enabled": source_model_re_enabled,
    }
