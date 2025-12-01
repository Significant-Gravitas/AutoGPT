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
    include = (
        {"Models": {"include": {"Costs": True}}}
        if include_models
        else None
    )
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

