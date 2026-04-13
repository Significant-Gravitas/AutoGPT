"""Database write operations for LLM registry admin API."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import prisma
import prisma.models

from backend.data import llm_registry
from backend.data.db import transaction
from backend.data.llm_registry.notifications import publish_registry_refresh_notification

logger = logging.getLogger(__name__)


def _node_model_value(slug: str) -> str:
    """Extract the model value stored in AgentNode.constantInput from a registry slug.

    Registry slugs are formatted as 'provider/model-name' (e.g. 'openai/gpt-4o').
    The LLM block stores only the model-name part (e.g. 'gpt-4o') in constantInput.
    """
    return slug.split("/", 1)[-1] if "/" in slug else slug


def _build_provider_data(
    name: str,
    display_name: str,
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provider data dict for Prisma operations."""
    return {
        "name": name,
        "displayName": display_name,
        "description": description,
        "defaultCredentialProvider": default_credential_provider,
        "defaultCredentialId": default_credential_id,
        "defaultCredentialType": default_credential_type,
        "metadata": prisma.Json(metadata or {}),
    }


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
    supports_tools: bool = False,
    supports_json_output: bool = False,
    supports_reasoning: bool = False,
    supports_parallel_tool_calls: bool = False,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build model data dict for Prisma operations."""
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
        "supportsTools": supports_tools,
        "supportsJsonOutput": supports_json_output,
        "supportsReasoning": supports_reasoning,
        "supportsParallelToolCalls": supports_parallel_tool_calls,
        "capabilities": prisma.Json(capabilities or {}),
        "metadata": prisma.Json(metadata or {}),
    }
    if creator_id:
        data["Creator"] = {"connect": {"id": creator_id}}
    return data


async def create_provider(
    name: str,
    display_name: str,
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmProvider:
    """Create a new LLM provider."""
    data = _build_provider_data(
        name=name,
        display_name=display_name,
        description=description,
        default_credential_provider=default_credential_provider,
        default_credential_id=default_credential_id,
        default_credential_type=default_credential_type,
        metadata=metadata,
    )
    provider = await prisma.models.LlmProvider.prisma().create(
        data=data,
        include={"Models": True},
    )
    if not provider:
        raise ValueError("Failed to create provider")
    return provider


async def update_provider(
    provider_id: str,
    display_name: str | None = None,
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmProvider:
    """Update an existing LLM provider."""
    # Fetch existing provider to get current name
    provider = await prisma.models.LlmProvider.prisma().find_unique(
        where={"id": provider_id}
    )
    if not provider:
        raise ValueError(f"Provider with id '{provider_id}' not found")

    # Build update data (only include fields that are provided)
    data: dict[str, Any] = {}
    if display_name is not None:
        data["displayName"] = display_name
    if description is not None:
        data["description"] = description
    if default_credential_provider is not None:
        data["defaultCredentialProvider"] = default_credential_provider
    if default_credential_id is not None:
        data["defaultCredentialId"] = default_credential_id
    if default_credential_type is not None:
        data["defaultCredentialType"] = default_credential_type
    if metadata is not None:
        data["metadata"] = prisma.Json(metadata)

    updated = await prisma.models.LlmProvider.prisma().update(
        where={"id": provider_id},
        data=data,
        include={"Models": True},
    )
    if not updated:
        raise ValueError("Failed to update provider")
    return updated


async def delete_provider(provider_id: str) -> bool:
    """Delete an LLM provider.

    A provider can only be deleted if it has no associated models.
    """
    # Check if provider exists
    provider = await prisma.models.LlmProvider.prisma().find_unique(
        where={"id": provider_id},
        include={"Models": True},
    )
    if not provider:
        raise ValueError(f"Provider with id '{provider_id}' not found")

    # Check if provider has any models
    model_count = len(provider.Models) if provider.Models else 0
    if model_count > 0:
        raise ValueError(
            f"Cannot delete provider '{provider.displayName}' because it has "
            f"{model_count} model(s). Delete all models first."
        )

    await prisma.models.LlmProvider.prisma().delete(where={"id": provider_id})
    return True


async def create_model(
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
    supports_tools: bool = False,
    supports_json_output: bool = False,
    supports_reasoning: bool = False,
    supports_parallel_tool_calls: bool = False,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmModel:
    """Create a new LLM model."""
    data = _build_model_data(
        slug=slug,
        display_name=display_name,
        provider_id=provider_id,
        context_window=context_window,
        price_tier=price_tier,
        description=description,
        creator_id=creator_id,
        max_output_tokens=max_output_tokens,
        is_enabled=is_enabled,
        is_recommended=is_recommended,
        supports_tools=supports_tools,
        supports_json_output=supports_json_output,
        supports_reasoning=supports_reasoning,
        supports_parallel_tool_calls=supports_parallel_tool_calls,
        capabilities=capabilities,
        metadata=metadata,
    )
    model = await prisma.models.LlmModel.prisma().create(
        data=data,
        include={"Costs": True, "Creator": True, "Provider": True},
    )
    if not model:
        raise ValueError("Failed to create model")
    return model


async def update_model(
    model_id: str,
    display_name: str | None = None,
    description: str | None = None,
    creator_id: str | None = None,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    price_tier: int | None = None,
    is_enabled: bool | None = None,
    is_recommended: bool | None = None,
    supports_tools: bool | None = None,
    supports_json_output: bool | None = None,
    supports_reasoning: bool | None = None,
    supports_parallel_tool_calls: bool | None = None,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmModel:
    """Update an existing LLM model.

    When is_recommended=True, clears the flag on all other models first so
    only one model can be recommended at a time.
    """
    # Build update data (only include fields that are provided)
    data: dict[str, Any] = {}
    if display_name is not None:
        data["displayName"] = display_name
    if description is not None:
        data["description"] = description
    if context_window is not None:
        data["contextWindow"] = context_window
    if max_output_tokens is not None:
        data["maxOutputTokens"] = max_output_tokens
    if price_tier is not None:
        data["priceTier"] = price_tier
    if is_enabled is not None:
        data["isEnabled"] = is_enabled
    if is_recommended is not None:
        data["isRecommended"] = is_recommended
    if supports_tools is not None:
        data["supportsTools"] = supports_tools
    if supports_json_output is not None:
        data["supportsJsonOutput"] = supports_json_output
    if supports_reasoning is not None:
        data["supportsReasoning"] = supports_reasoning
    if supports_parallel_tool_calls is not None:
        data["supportsParallelToolCalls"] = supports_parallel_tool_calls
    if capabilities is not None:
        data["capabilities"] = prisma.Json(capabilities)
    if metadata is not None:
        data["metadata"] = prisma.Json(metadata)
    if creator_id is not None:
        data["creatorId"] = creator_id if creator_id else None

    async with transaction() as tx:
        # Enforce single recommended model: unset all others first.
        if is_recommended is True:
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
    import prisma as prisma_module

    model_value = _node_model_value(slug)
    count_result = await prisma_module.get_client().query_raw(
        """
        SELECT COUNT(*) as count
        FROM "AgentNode"
        WHERE "constantInput"::jsonb->>'model' = $1
        """,
        model_value,
    )
    node_count = int(count_result[0]["count"]) if count_result else 0
    return {"model_slug": slug, "node_count": node_count}


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

    nodes_migrated = 0
    migration_id: str | None = None

    if not is_enabled and migrate_to_slug:
        async with transaction() as tx:
            replacement = await tx.llmmodel.find_unique(
                where={"slug": migrate_to_slug}
            )
            if not replacement:
                raise ValueError(
                    f"Replacement model '{migrate_to_slug}' not found"
                )
            if not replacement.isEnabled:
                raise ValueError(
                    f"Replacement model '{migrate_to_slug}' is disabled. "
                    f"Please enable it before using it as a replacement."
                )

            source_value = _node_model_value(model.slug)
            target_value = _node_model_value(migrate_to_slug)
            node_ids_result = await tx.query_raw(
                """
                SELECT id
                FROM "AgentNode"
                WHERE "constantInput"::jsonb->>'model' = $1
                FOR UPDATE
                """,
                source_value,
            )
            migrated_node_ids = (
                [row["id"] for row in node_ids_result] if node_ids_result else []
            )
            nodes_migrated = len(migrated_node_ids)

            if nodes_migrated > 0:
                node_ids_json = json.dumps(migrated_node_ids)
                await tx.execute_raw(
                    """
                    UPDATE "AgentNode"
                    SET "constantInput" = JSONB_SET(
                        "constantInput"::jsonb,
                        '{model}',
                        to_jsonb($1::text)
                    )
                    WHERE id::text IN (
                        SELECT jsonb_array_elements_text($2::jsonb)
                    )
                    """,
                    target_value,
                    node_ids_json,
                )

            await tx.llmmodel.update(
                where={"id": model_id},
                data={"isEnabled": is_enabled},
            )

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
            where={"id": model_id},
            data={"isEnabled": is_enabled},
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

    If workflows are using this model and no replacement is given, raises ValueError.
    If replacement is given, atomically migrates all affected nodes then deletes.
    """
    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}, include={"Costs": True}
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    deleted_slug = model.slug
    deleted_display_name = model.displayName

    async with transaction() as tx:
        count_result = await tx.query_raw(
            """
            SELECT COUNT(*) as count
            FROM "AgentNode"
            WHERE "constantInput"::jsonb->>'model' = $1
            """,
            deleted_slug,
        )
        nodes_to_migrate = int(count_result[0]["count"]) if count_result else 0

        if nodes_to_migrate > 0:
            if not replacement_model_slug:
                raise ValueError(
                    f"Cannot delete model '{deleted_slug}': {nodes_to_migrate} workflow node(s) "
                    f"are using it. Please provide a replacement_model_slug to migrate them."
                )
            replacement = await tx.llmmodel.find_unique(
                where={"slug": replacement_model_slug}
            )
            if not replacement:
                raise ValueError(
                    f"Replacement model '{replacement_model_slug}' not found"
                )
            if not replacement.isEnabled:
                raise ValueError(
                    f"Replacement model '{replacement_model_slug}' is disabled."
                )

            await tx.execute_raw(
                """
                UPDATE "AgentNode"
                SET "constantInput" = JSONB_SET(
                    "constantInput"::jsonb,
                    '{model}',
                    to_jsonb($1::text)
                )
                WHERE "constantInput"::jsonb->>'model' = $2
                """,
                replacement_model_slug,
                deleted_slug,
            )

        await tx.llmmodel.delete(where={"id": model_id})

    return {
        "deleted_model_slug": deleted_slug,
        "deleted_model_display_name": deleted_display_name,
        "replacement_model_slug": replacement_model_slug,
        "nodes_migrated": nodes_to_migrate,
    }


async def list_migrations(
    include_reverted: bool = False,
) -> list[dict[str, Any]]:
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
        raise ValueError(
            f"Migration '{migration_id}' has already been reverted"
        )

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
        else json.loads(migration.migratedNodeIds)  # type: ignore
    )
    if not migrated_node_ids:
        raise ValueError("No nodes to revert in this migration")

    source_model_re_enabled = False

    async with transaction() as tx:
        if not source_model.isEnabled and re_enable_source_model:
            await tx.llmmodel.update(
                where={"id": source_model.id},
                data={"isEnabled": True},
            )
            source_model_re_enabled = True

        node_ids_json = json.dumps(migrated_node_ids)
        result = await tx.execute_raw(
            """
            UPDATE "AgentNode"
            SET "constantInput" = JSONB_SET(
                "constantInput"::jsonb,
                '{model}',
                to_jsonb($1::text)
            )
            WHERE id::text IN (
                SELECT jsonb_array_elements_text($2::jsonb)
            )
            AND "constantInput"::jsonb->>'model' = $3
            """,
            migration.sourceModelSlug,
            node_ids_json,
            migration.targetModelSlug,
        )
        nodes_reverted = result if isinstance(result, int) else 0

        await tx.llmmodelmigration.update(
            where={"id": migration_id},
            data={
                "isReverted": True,
                "revertedAt": datetime.now(timezone.utc),
            },
        )

    return {
        "migration_id": migration_id,
        "source_model_slug": migration.sourceModelSlug,
        "target_model_slug": migration.targetModelSlug,
        "nodes_reverted": nodes_reverted,
        "nodes_already_changed": len(migrated_node_ids) - nodes_reverted,
        "source_model_re_enabled": source_model_re_enabled,
    }


async def refresh_runtime_caches() -> None:
    llm_registry.clear_registry_cache()
    await llm_registry.refresh_llm_registry()
    await publish_registry_refresh_notification()
