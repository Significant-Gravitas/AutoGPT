from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import prisma
import prisma.models

from backend.data.db import transaction
from backend.server.v2.llm import model as llm_model
from backend.util.models import Pagination


def _json_dict(value: Any | None) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _map_cost(record: prisma.models.LlmModelCost) -> llm_model.LlmModelCost:
    return llm_model.LlmModelCost(
        id=record.id,
        unit=record.unit,
        credit_cost=record.creditCost,
        credential_provider=record.credentialProvider,
        credential_id=record.credentialId,
        credential_type=record.credentialType,
        currency=record.currency,
        metadata=_json_dict(record.metadata),
    )


def _map_creator(
    record: prisma.models.LlmModelCreator,
) -> llm_model.LlmModelCreator:
    return llm_model.LlmModelCreator(
        id=record.id,
        name=record.name,
        display_name=record.displayName,
        description=record.description,
        website_url=record.websiteUrl,
        logo_url=record.logoUrl,
        metadata=_json_dict(record.metadata),
    )


def _map_model(record: prisma.models.LlmModel) -> llm_model.LlmModel:
    costs = []
    if record.Costs:
        costs = [_map_cost(cost) for cost in record.Costs]

    creator = None
    if hasattr(record, "Creator") and record.Creator:
        creator = _map_creator(record.Creator)

    return llm_model.LlmModel(
        id=record.id,
        slug=record.slug,
        display_name=record.displayName,
        description=record.description,
        provider_id=record.providerId,
        creator_id=record.creatorId,
        creator=creator,
        context_window=record.contextWindow,
        max_output_tokens=record.maxOutputTokens,
        is_enabled=record.isEnabled,
        is_recommended=record.isRecommended,
        capabilities=_json_dict(record.capabilities),
        metadata=_json_dict(record.metadata),
        costs=costs,
    )


def _map_provider(record: prisma.models.LlmProvider) -> llm_model.LlmProvider:
    models: list[llm_model.LlmModel] = []
    if record.Models:
        models = [_map_model(model) for model in record.Models]

    return llm_model.LlmProvider(
        id=record.id,
        name=record.name,
        display_name=record.displayName,
        description=record.description,
        default_credential_provider=record.defaultCredentialProvider,
        default_credential_id=record.defaultCredentialId,
        default_credential_type=record.defaultCredentialType,
        supports_tools=record.supportsTools,
        supports_json_output=record.supportsJsonOutput,
        supports_reasoning=record.supportsReasoning,
        supports_parallel_tool=record.supportsParallelTool,
        metadata=_json_dict(record.metadata),
        models=models,
    )


async def list_providers(
    include_models: bool = True, enabled_only: bool = False
) -> list[llm_model.LlmProvider]:
    """
    List all LLM providers.

    Args:
        include_models: Whether to include models for each provider
        enabled_only: If True, only include enabled models (for public routes)
    """
    include: Any = None
    if include_models:
        model_where = {"isEnabled": True} if enabled_only else None
        include = {
            "Models": {
                "include": {"Costs": True, "Creator": True},
                "where": model_where,
            }
        }
    records = await prisma.models.LlmProvider.prisma().find_many(include=include)
    return [_map_provider(record) for record in records]


async def upsert_provider(
    request: llm_model.UpsertLlmProviderRequest,
    provider_id: str | None = None,
) -> llm_model.LlmProvider:
    data: Any = {
        "name": request.name,
        "displayName": request.display_name,
        "description": request.description,
        "defaultCredentialProvider": request.default_credential_provider,
        "defaultCredentialId": request.default_credential_id,
        "defaultCredentialType": request.default_credential_type,
        "supportsTools": request.supports_tools,
        "supportsJsonOutput": request.supports_json_output,
        "supportsReasoning": request.supports_reasoning,
        "supportsParallelTool": request.supports_parallel_tool,
        "metadata": prisma.Json(request.metadata or {}),
    }
    include: Any = {"Models": {"include": {"Costs": True, "Creator": True}}}
    if provider_id:
        record = await prisma.models.LlmProvider.prisma().update(
            where={"id": provider_id},
            data=data,
            include=include,
        )
    else:
        record = await prisma.models.LlmProvider.prisma().create(
            data=data,
            include=include,
        )
    if record is None:
        raise ValueError("Failed to create/update provider")
    return _map_provider(record)


async def delete_provider(provider_id: str) -> bool:
    """
    Delete an LLM provider.

    A provider can only be deleted if it has no associated models.
    Due to onDelete: Restrict on LlmModel.Provider, the database will
    block deletion if models exist.

    Args:
        provider_id: UUID of the provider to delete

    Returns:
        True if deleted successfully

    Raises:
        ValueError: If provider not found or has associated models
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

    # Safe to delete
    await prisma.models.LlmProvider.prisma().delete(where={"id": provider_id})
    return True


async def list_models(
    provider_id: str | None = None,
    enabled_only: bool = False,
    page: int = 1,
    page_size: int = 50,
) -> llm_model.LlmModelsResponse:
    """
    List LLM models with pagination.

    Args:
        provider_id: Optional filter by provider ID
        enabled_only: If True, only return enabled models (for public routes)
        page: Page number (1-indexed)
        page_size: Number of models per page
    """
    where: Any = {}
    if provider_id:
        where["providerId"] = provider_id
    if enabled_only:
        where["isEnabled"] = True

    # Get total count for pagination
    total_items = await prisma.models.LlmModel.prisma().count(
        where=where if where else None
    )

    # Calculate pagination
    skip = (page - 1) * page_size
    total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 0

    records = await prisma.models.LlmModel.prisma().find_many(
        where=where if where else None,
        include={"Costs": True, "Creator": True},
        skip=skip,
        take=page_size,
    )
    models = [_map_model(record) for record in records]

    return llm_model.LlmModelsResponse(
        models=models,
        pagination=Pagination(
            total_items=total_items,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )


def _cost_create_payload(
    costs: Sequence[llm_model.LlmModelCostInput],
) -> dict[str, Iterable[dict[str, Any]]]:

    create_items = []
    for cost in costs:
        item: dict[str, Any] = {
            "unit": cost.unit,
            "creditCost": cost.credit_cost,
            "credentialProvider": cost.credential_provider,
        }
        # Only include optional fields if they have values
        if cost.credential_id:
            item["credentialId"] = cost.credential_id
        if cost.credential_type:
            item["credentialType"] = cost.credential_type
        if cost.currency:
            item["currency"] = cost.currency
        # Handle metadata - use Prisma Json type
        if cost.metadata is not None and cost.metadata != {}:
            item["metadata"] = prisma.Json(cost.metadata)
        create_items.append(item)
    return {"create": create_items}


async def create_model(
    request: llm_model.CreateLlmModelRequest,
) -> llm_model.LlmModel:
    data: Any = {
        "slug": request.slug,
        "displayName": request.display_name,
        "description": request.description,
        "Provider": {"connect": {"id": request.provider_id}},
        "contextWindow": request.context_window,
        "maxOutputTokens": request.max_output_tokens,
        "isEnabled": request.is_enabled,
        "capabilities": prisma.Json(request.capabilities or {}),
        "metadata": prisma.Json(request.metadata or {}),
        "Costs": _cost_create_payload(request.costs),
    }
    if request.creator_id:
        data["Creator"] = {"connect": {"id": request.creator_id}}

    record = await prisma.models.LlmModel.prisma().create(
        data=data,
        include={"Costs": True, "Creator": True, "Provider": True},
    )
    return _map_model(record)


async def update_model(
    model_id: str,
    request: llm_model.UpdateLlmModelRequest,
) -> llm_model.LlmModel:
    # Build scalar field updates (non-relation fields)
    scalar_data: Any = {}
    if request.display_name is not None:
        scalar_data["displayName"] = request.display_name
    if request.description is not None:
        scalar_data["description"] = request.description
    if request.context_window is not None:
        scalar_data["contextWindow"] = request.context_window
    if request.max_output_tokens is not None:
        scalar_data["maxOutputTokens"] = request.max_output_tokens
    if request.is_enabled is not None:
        scalar_data["isEnabled"] = request.is_enabled
    if request.capabilities is not None:
        scalar_data["capabilities"] = request.capabilities
    if request.metadata is not None:
        scalar_data["metadata"] = request.metadata
    # Foreign keys can be updated directly as scalar fields
    if request.provider_id is not None:
        scalar_data["providerId"] = request.provider_id
    if request.creator_id is not None:
        # Empty string means remove the creator
        scalar_data["creatorId"] = request.creator_id if request.creator_id else None

    # If we have costs to update, we need to handle them separately
    # because nested writes have different constraints
    if request.costs is not None:
        # Wrap cost replacement in a transaction for atomicity
        async with transaction() as tx:
            # First update scalar fields
            if scalar_data:
                await tx.llmmodel.update(
                    where={"id": model_id},
                    data=scalar_data,
                )
            # Then handle costs: delete existing and create new
            await tx.llmmodelcost.delete_many(where={"llmModelId": model_id})
            if request.costs:
                cost_payload = _cost_create_payload(request.costs)
                for cost_item in cost_payload["create"]:
                    cost_item["llmModelId"] = model_id
                    await tx.llmmodelcost.create(data=cast(Any, cost_item))
        # Fetch the updated record (outside transaction)
        record = await prisma.models.LlmModel.prisma().find_unique(
            where={"id": model_id},
            include={"Costs": True, "Creator": True},
        )
    else:
        # No costs update - simple update
        record = await prisma.models.LlmModel.prisma().update(
            where={"id": model_id},
            data=scalar_data,
            include={"Costs": True, "Creator": True},
        )

    if not record:
        raise ValueError(f"Model with id '{model_id}' not found")
    return _map_model(record)


async def toggle_model(
    model_id: str,
    is_enabled: bool,
    migrate_to_slug: str | None = None,
    migration_reason: str | None = None,
    custom_credit_cost: int | None = None,
) -> llm_model.ToggleLlmModelResponse:
    """
    Toggle a model's enabled status, optionally migrating workflows when disabling.

    Args:
        model_id: UUID of the model to toggle
        is_enabled: New enabled status
        migrate_to_slug: If disabling and this is provided, migrate all workflows
                         using this model to the specified replacement model
        migration_reason: Optional reason for the migration (e.g., "Provider outage")
        custom_credit_cost: Optional custom pricing override for migrated workflows.
                           When set, the billing system should use this cost instead
                           of the target model's cost for affected nodes.

    Returns:
        ToggleLlmModelResponse with the updated model and optional migration stats
    """
    import json

    # Get the model being toggled
    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}, include={"Costs": True}
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    nodes_migrated = 0
    migration_id: str | None = None

    # If disabling with migration, perform migration first
    if not is_enabled and migrate_to_slug:
        # Validate replacement model exists and is enabled
        replacement = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": migrate_to_slug}
        )
        if not replacement:
            raise ValueError(f"Replacement model '{migrate_to_slug}' not found")
        if not replacement.isEnabled:
            raise ValueError(
                f"Replacement model '{migrate_to_slug}' is disabled. "
                f"Please enable it before using it as a replacement."
            )

        # Perform all operations atomically within a single transaction
        # This ensures no nodes are missed between query and update
        async with transaction() as tx:
            # Get the IDs of nodes that will be migrated (inside transaction for consistency)
            node_ids_result = await tx.query_raw(
                """
                SELECT id
                FROM "AgentNode"
                WHERE "constantInput"::jsonb->>'model' = $1
                FOR UPDATE
                """,
                model.slug,
            )
            migrated_node_ids = (
                [row["id"] for row in node_ids_result] if node_ids_result else []
            )
            nodes_migrated = len(migrated_node_ids)

            if nodes_migrated > 0:
                # Update by IDs to ensure we only update the exact nodes we queried
                # Use JSON array and jsonb_array_elements_text for safe parameterization
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
                    migrate_to_slug,
                    node_ids_json,
                )

            record = await tx.llmmodel.update(
                where={"id": model_id},
                data={"isEnabled": is_enabled},
                include={"Costs": True},
            )

            # Create migration record for revert capability
            if nodes_migrated > 0:
                migration_data: Any = {
                    "sourceModelSlug": model.slug,
                    "targetModelSlug": migrate_to_slug,
                    "reason": migration_reason,
                    "migratedNodeIds": json.dumps(migrated_node_ids),
                    "nodeCount": nodes_migrated,
                    "customCreditCost": custom_credit_cost,
                }
                migration_record = await tx.llmmodelmigration.create(
                    data=migration_data
                )
                migration_id = migration_record.id
    else:
        # Simple toggle without migration
        record = await prisma.models.LlmModel.prisma().update(
            where={"id": model_id},
            data={"isEnabled": is_enabled},
            include={"Costs": True},
        )

    if record is None:
        raise ValueError(f"Model with id '{model_id}' not found")
    return llm_model.ToggleLlmModelResponse(
        model=_map_model(record),
        nodes_migrated=nodes_migrated,
        migrated_to_slug=migrate_to_slug if nodes_migrated > 0 else None,
        migration_id=migration_id,
    )


async def get_model_usage(model_id: str) -> llm_model.LlmModelUsageResponse:
    """Get usage count for a model."""
    import prisma as prisma_module

    model = await prisma.models.LlmModel.prisma().find_unique(where={"id": model_id})
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    count_result = await prisma_module.get_client().query_raw(
        """
        SELECT COUNT(*) as count
        FROM "AgentNode"
        WHERE "constantInput"::jsonb->>'model' = $1
        """,
        model.slug,
    )
    node_count = int(count_result[0]["count"]) if count_result else 0

    return llm_model.LlmModelUsageResponse(model_slug=model.slug, node_count=node_count)


async def delete_model(
    model_id: str, replacement_model_slug: str | None = None
) -> llm_model.DeleteLlmModelResponse:
    """
    Delete a model and optionally migrate all AgentNodes using it to a replacement model.

    This performs an atomic operation within a database transaction:
    1. Validates the model exists
    2. Counts affected nodes
    3. If nodes exist, validates replacement model and migrates them
    4. Deletes the LlmModel record (CASCADE deletes costs)

    Args:
        model_id: UUID of the model to delete
        replacement_model_slug: Slug of the model to migrate to (required only if nodes use this model)

    Returns:
        DeleteLlmModelResponse with migration stats

    Raises:
        ValueError: If model not found, nodes exist but no replacement provided,
                    replacement not found, or replacement is disabled
    """
    # 1. Get the model being deleted (validation - outside transaction)
    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}, include={"Costs": True}
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    deleted_slug = model.slug
    deleted_display_name = model.displayName

    # 2. Count affected nodes first to determine if replacement is needed
    import prisma as prisma_module

    count_result = await prisma_module.get_client().query_raw(
        """
        SELECT COUNT(*) as count
        FROM "AgentNode"
        WHERE "constantInput"::jsonb->>'model' = $1
        """,
        deleted_slug,
    )
    nodes_to_migrate = int(count_result[0]["count"]) if count_result else 0

    # 3. Validate replacement model only if there are nodes to migrate
    if nodes_to_migrate > 0:
        if not replacement_model_slug:
            raise ValueError(
                f"Cannot delete model '{deleted_slug}': {nodes_to_migrate} workflow node(s) "
                f"are using it. Please provide a replacement_model_slug to migrate them."
            )
        replacement = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": replacement_model_slug}
        )
        if not replacement:
            raise ValueError(f"Replacement model '{replacement_model_slug}' not found")
        if not replacement.isEnabled:
            raise ValueError(
                f"Replacement model '{replacement_model_slug}' is disabled. "
                f"Please enable it before using it as a replacement."
            )

    # 4. Perform migration (if needed) and deletion atomically within a transaction
    async with transaction() as tx:
        # Migrate all AgentNode.constantInput->model to replacement
        if nodes_to_migrate > 0 and replacement_model_slug:
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

        # Delete the model (CASCADE will delete costs automatically)
        await tx.llmmodel.delete(where={"id": model_id})

    # Build appropriate message based on whether migration happened
    if nodes_to_migrate > 0:
        message = (
            f"Successfully deleted model '{deleted_display_name}' ({deleted_slug}) "
            f"and migrated {nodes_to_migrate} workflow node(s) to '{replacement_model_slug}'."
        )
    else:
        message = (
            f"Successfully deleted model '{deleted_display_name}' ({deleted_slug}). "
            f"No workflows were using this model."
        )

    return llm_model.DeleteLlmModelResponse(
        deleted_model_slug=deleted_slug,
        deleted_model_display_name=deleted_display_name,
        replacement_model_slug=replacement_model_slug,
        nodes_migrated=nodes_to_migrate,
        message=message,
    )


def _map_migration(
    record: prisma.models.LlmModelMigration,
) -> llm_model.LlmModelMigration:
    return llm_model.LlmModelMigration(
        id=record.id,
        source_model_slug=record.sourceModelSlug,
        target_model_slug=record.targetModelSlug,
        reason=record.reason,
        node_count=record.nodeCount,
        custom_credit_cost=record.customCreditCost,
        is_reverted=record.isReverted,
        created_at=record.createdAt,
        reverted_at=record.revertedAt,
    )


async def list_migrations(
    include_reverted: bool = False,
) -> list[llm_model.LlmModelMigration]:
    """
    List model migrations, optionally including reverted ones.

    Args:
        include_reverted: If True, include reverted migrations. Default is False.

    Returns:
        List of LlmModelMigration records
    """
    where: Any = None if include_reverted else {"isReverted": False}
    records = await prisma.models.LlmModelMigration.prisma().find_many(
        where=where,
        order={"createdAt": "desc"},
    )
    return [_map_migration(record) for record in records]


async def get_migration(migration_id: str) -> llm_model.LlmModelMigration | None:
    """Get a specific migration by ID."""
    record = await prisma.models.LlmModelMigration.prisma().find_unique(
        where={"id": migration_id}
    )
    return _map_migration(record) if record else None


async def revert_migration(
    migration_id: str,
    re_enable_source_model: bool = True,
) -> llm_model.RevertMigrationResponse:
    """
    Revert a model migration, restoring affected nodes to their original model.

    This only reverts the specific nodes that were migrated, not all nodes
    currently using the target model.

    Args:
        migration_id: UUID of the migration to revert
        re_enable_source_model: Whether to re-enable the source model if it's disabled

    Returns:
        RevertMigrationResponse with revert stats

    Raises:
        ValueError: If migration not found, already reverted, or source model not available
    """
    import json
    from datetime import datetime, timezone

    # Get the migration record
    migration = await prisma.models.LlmModelMigration.prisma().find_unique(
        where={"id": migration_id}
    )
    if not migration:
        raise ValueError(f"Migration with id '{migration_id}' not found")

    if migration.isReverted:
        raise ValueError(
            f"Migration '{migration_id}' has already been reverted "
            f"on {migration.revertedAt.isoformat() if migration.revertedAt else 'unknown date'}"
        )

    # Check if source model exists
    source_model = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": migration.sourceModelSlug}
    )
    if not source_model:
        raise ValueError(
            f"Source model '{migration.sourceModelSlug}' no longer exists. "
            f"Cannot revert migration."
        )

    # Get the migrated node IDs (Prisma auto-parses JSONB to list)
    migrated_node_ids: list[str] = (
        migration.migratedNodeIds
        if isinstance(migration.migratedNodeIds, list)
        else json.loads(migration.migratedNodeIds)  # type: ignore
    )
    if not migrated_node_ids:
        raise ValueError("No nodes to revert in this migration")

    # Track if we need to re-enable the source model
    source_model_was_disabled = not source_model.isEnabled
    should_re_enable = source_model_was_disabled and re_enable_source_model
    source_model_re_enabled = False

    # Perform revert atomically
    async with transaction() as tx:
        # Re-enable the source model if requested and it was disabled
        if should_re_enable:
            await tx.llmmodel.update(
                where={"id": source_model.id},
                data={"isEnabled": True},
            )
            source_model_re_enabled = True

        # Update only the specific nodes that were migrated
        # We need to check that they still have the target model (haven't been changed since)
        # Use a single batch update for efficiency
        # Use JSON array and jsonb_array_elements_text for safe parameterization
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
        nodes_reverted = result if result else 0

        # Mark migration as reverted
        await tx.llmmodelmigration.update(
            where={"id": migration_id},
            data={
                "isReverted": True,
                "revertedAt": datetime.now(timezone.utc),
            },
        )

    # Calculate nodes that were already changed since migration
    nodes_already_changed = len(migrated_node_ids) - nodes_reverted

    # Build appropriate message
    message_parts = [
        f"Successfully reverted migration: {nodes_reverted} node(s) restored "
        f"from '{migration.targetModelSlug}' to '{migration.sourceModelSlug}'."
    ]
    if nodes_already_changed > 0:
        message_parts.append(
            f" {nodes_already_changed} node(s) were already changed and not reverted."
        )
    if source_model_re_enabled:
        message_parts.append(
            f" Model '{migration.sourceModelSlug}' has been re-enabled."
        )

    return llm_model.RevertMigrationResponse(
        migration_id=migration_id,
        source_model_slug=migration.sourceModelSlug,
        target_model_slug=migration.targetModelSlug,
        nodes_reverted=nodes_reverted,
        nodes_already_changed=nodes_already_changed,
        source_model_re_enabled=source_model_re_enabled,
        message="".join(message_parts),
    )


# ============================================================================
# Creator CRUD operations
# ============================================================================


async def list_creators() -> list[llm_model.LlmModelCreator]:
    """List all LLM model creators."""
    records = await prisma.models.LlmModelCreator.prisma().find_many(
        order={"displayName": "asc"}
    )
    return [_map_creator(record) for record in records]


async def get_creator(creator_id: str) -> llm_model.LlmModelCreator | None:
    """Get a specific creator by ID."""
    record = await prisma.models.LlmModelCreator.prisma().find_unique(
        where={"id": creator_id}
    )
    return _map_creator(record) if record else None


async def upsert_creator(
    request: llm_model.UpsertLlmCreatorRequest,
    creator_id: str | None = None,
) -> llm_model.LlmModelCreator:
    """Create or update a model creator."""
    data: Any = {
        "name": request.name,
        "displayName": request.display_name,
        "description": request.description,
        "websiteUrl": request.website_url,
        "logoUrl": request.logo_url,
        "metadata": prisma.Json(request.metadata or {}),
    }
    if creator_id:
        record = await prisma.models.LlmModelCreator.prisma().update(
            where={"id": creator_id},
            data=data,
        )
    else:
        record = await prisma.models.LlmModelCreator.prisma().create(data=data)
    if record is None:
        raise ValueError("Failed to create/update creator")
    return _map_creator(record)


async def delete_creator(creator_id: str) -> bool:
    """
    Delete a model creator.

    This will set creatorId to NULL on all associated models (due to onDelete: SetNull).

    Args:
        creator_id: UUID of the creator to delete

    Returns:
        True if deleted successfully

    Raises:
        ValueError: If creator not found
    """
    creator = await prisma.models.LlmModelCreator.prisma().find_unique(
        where={"id": creator_id}
    )
    if not creator:
        raise ValueError(f"Creator with id '{creator_id}' not found")

    await prisma.models.LlmModelCreator.prisma().delete(where={"id": creator_id})
    return True


async def get_recommended_model() -> llm_model.LlmModel | None:
    """
    Get the currently recommended LLM model.

    Returns:
        The recommended model, or None if no model is marked as recommended.
    """
    record = await prisma.models.LlmModel.prisma().find_first(
        where={"isRecommended": True, "isEnabled": True},
        include={"Costs": True, "Creator": True},
    )
    return _map_model(record) if record else None


async def set_recommended_model(
    model_id: str,
) -> tuple[llm_model.LlmModel, str | None]:
    """
    Set a model as the recommended model.

    This will clear the isRecommended flag from any other model and set it
    on the specified model. The model must be enabled.

    Args:
        model_id: UUID of the model to set as recommended

    Returns:
        Tuple of (the updated model, previous recommended model slug or None)

    Raises:
        ValueError: If model not found or not enabled
    """
    # First, verify the model exists and is enabled
    target_model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}
    )
    if not target_model:
        raise ValueError(f"Model with id '{model_id}' not found")
    if not target_model.isEnabled:
        raise ValueError(
            f"Cannot set disabled model '{target_model.slug}' as recommended"
        )

    # Get the current recommended model (if any)
    current_recommended = await prisma.models.LlmModel.prisma().find_first(
        where={"isRecommended": True}
    )
    previous_slug = current_recommended.slug if current_recommended else None

    # Use a transaction to ensure atomicity
    async with transaction() as tx:
        # Clear isRecommended from all models
        await tx.llmmodel.update_many(
            where={"isRecommended": True},
            data={"isRecommended": False},
        )
        # Set the new recommended model
        await tx.llmmodel.update(
            where={"id": model_id},
            data={"isRecommended": True},
        )

    # Fetch and return the updated model
    updated_record = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id},
        include={"Costs": True, "Creator": True},
    )
    if not updated_record:
        raise ValueError("Failed to fetch updated model")

    return _map_model(updated_record), previous_slug


async def get_recommended_model_slug() -> str | None:
    """
    Get the slug of the currently recommended LLM model.

    Returns:
        The slug of the recommended model, or None if no model is marked as recommended.
    """
    record = await prisma.models.LlmModel.prisma().find_first(
        where={"isRecommended": True, "isEnabled": True},
    )
    return record.slug if record else None
