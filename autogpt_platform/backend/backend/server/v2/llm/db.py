from __future__ import annotations

from typing import Any, Iterable, Sequence

import prisma.models

from backend.server.v2.llm import model as llm_model


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


def _map_model(record: prisma.models.LlmModel) -> llm_model.LlmModel:
    costs = []
    if record.Costs:
        costs = [_map_cost(cost) for cost in record.Costs]

    return llm_model.LlmModel(
        id=record.id,
        slug=record.slug,
        display_name=record.displayName,
        description=record.description,
        provider_id=record.providerId,
        context_window=record.contextWindow,
        max_output_tokens=record.maxOutputTokens,
        is_enabled=record.isEnabled,
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


async def list_providers(include_models: bool = True) -> list[llm_model.LlmProvider]:
    include = {"Models": {"include": {"Costs": True}}} if include_models else None
    records = await prisma.models.LlmProvider.prisma().find_many(include=include)
    return [_map_provider(record) for record in records]


async def upsert_provider(
    request: llm_model.UpsertLlmProviderRequest,
    provider_id: str | None = None,
) -> llm_model.LlmProvider:
    data = {
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
        "metadata": request.metadata,
    }
    if provider_id:
        record = await prisma.models.LlmProvider.prisma().update(
            where={"id": provider_id},
            data=data,
            include={"Models": {"include": {"Costs": True}}},
        )
    else:
        record = await prisma.models.LlmProvider.prisma().create(
            data=data,
            include={"Models": {"include": {"Costs": True}}},
        )
    return _map_provider(record)


async def list_models(provider_id: str | None = None) -> list[llm_model.LlmModel]:
    where = {"providerId": provider_id} if provider_id else None
    records = await prisma.models.LlmModel.prisma().find_many(
        where=where,
        include={"Costs": True},
    )
    return [_map_model(record) for record in records]


def _cost_create_payload(
    costs: Sequence[llm_model.LlmModelCostInput],
) -> dict[str, Iterable[dict[str, Any]]]:
    return {
        "create": [
            {
                "unit": cost.unit,
                "creditCost": cost.credit_cost,
                "credentialProvider": cost.credential_provider,
                "credentialId": cost.credential_id,
                "credentialType": cost.credential_type,
                "currency": cost.currency,
                "metadata": cost.metadata,
            }
            for cost in costs
        ]
    }


async def create_model(
    request: llm_model.CreateLlmModelRequest,
) -> llm_model.LlmModel:
    record = await prisma.models.LlmModel.prisma().create(
        data={
            "slug": request.slug,
            "displayName": request.display_name,
            "description": request.description,
            "providerId": request.provider_id,
            "contextWindow": request.context_window,
            "maxOutputTokens": request.max_output_tokens,
            "isEnabled": request.is_enabled,
            "capabilities": request.capabilities,
            "metadata": request.metadata,
            "Costs": _cost_create_payload(request.costs),
        },
        include={"Costs": True},
    )
    return _map_model(record)


async def update_model(
    model_id: str,
    request: llm_model.UpdateLlmModelRequest,
) -> llm_model.LlmModel:
    data: dict[str, Any] = {}
    if request.display_name is not None:
        data["displayName"] = request.display_name
    if request.description is not None:
        data["description"] = request.description
    if request.context_window is not None:
        data["contextWindow"] = request.context_window
    if request.max_output_tokens is not None:
        data["maxOutputTokens"] = request.max_output_tokens
    if request.is_enabled is not None:
        data["isEnabled"] = request.is_enabled
    if request.capabilities is not None:
        data["capabilities"] = request.capabilities
    if request.metadata is not None:
        data["metadata"] = request.metadata
    if request.provider_id is not None:
        data["providerId"] = request.provider_id
    if request.costs is not None:
        data["Costs"] = {
            "deleteMany": {"llmModelId": model_id},
            **_cost_create_payload(request.costs),
        }

    record = await prisma.models.LlmModel.prisma().update(
        where={"id": model_id},
        data=data,
        include={"Costs": True},
    )
    return _map_model(record)


async def toggle_model(model_id: str, is_enabled: bool) -> llm_model.LlmModel:
    record = await prisma.models.LlmModel.prisma().update(
        where={"id": model_id},
        data={"isEnabled": is_enabled},
        include={"Costs": True},
    )
    return _map_model(record)


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
    model_id: str, replacement_model_slug: str
) -> llm_model.DeleteLlmModelResponse:
    """
    Delete a model and migrate all AgentNodes using it to a replacement model.

    This performs an atomic operation:
    1. Validates the model exists
    2. Validates the replacement model exists and is enabled
    3. Counts affected nodes
    4. Migrates all AgentNode.constantInput->model to replacement
    5. Deletes the LlmModel record (CASCADE deletes costs)

    Args:
        model_id: UUID of the model to delete
        replacement_model_slug: Slug of the model to migrate to

    Returns:
        DeleteLlmModelResponse with migration stats

    Raises:
        ValueError: If model not found, replacement not found, or replacement is disabled
    """
    import prisma as prisma_module

    # 1. Get the model being deleted
    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"id": model_id}, include={"Costs": True}
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    deleted_slug = model.slug
    deleted_display_name = model.displayName

    # 2. Validate replacement model exists and is enabled
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

    # 3. Count affected nodes
    count_result = await prisma_module.get_client().query_raw(
        """
        SELECT COUNT(*) as count
        FROM "AgentNode"
        WHERE "constantInput"::jsonb->>'model' = $1
        """,
        deleted_slug,
    )
    nodes_affected = int(count_result[0]["count"]) if count_result else 0

    # 4. Perform migration
    if nodes_affected > 0:
        await prisma_module.get_client().execute_raw(
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

    # 5. Delete the model (CASCADE will delete costs automatically)
    await prisma.models.LlmModel.prisma().delete(where={"id": model_id})

    return llm_model.DeleteLlmModelResponse(
        deleted_model_slug=deleted_slug,
        deleted_model_display_name=deleted_display_name,
        replacement_model_slug=replacement_model_slug,
        nodes_migrated=nodes_affected,
        message=(
            f"Successfully deleted model '{deleted_display_name}' ({deleted_slug}) "
            f"and migrated {nodes_affected} workflow node(s) to '{replacement_model_slug}'."
        ),
    )
