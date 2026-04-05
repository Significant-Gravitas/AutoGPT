"""Public read-only API for LLM registry."""

import autogpt_libs.auth
import fastapi

from backend.data.llm_registry import (
    RegistryModelCreator,
    get_all_models,
    get_enabled_models,
)
from backend.server.v2.llm import model as llm_model

router = fastapi.APIRouter(
    prefix="/llm",
    tags=["llm"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
)


def _map_creator(
    creator: RegistryModelCreator | None,
) -> llm_model.LlmModelCreator | None:
    """Convert registry creator to API model."""
    if not creator:
        return None
    return llm_model.LlmModelCreator(
        id=creator.id,
        name=creator.name,
        display_name=creator.display_name,
        description=creator.description,
        website_url=creator.website_url,
        logo_url=creator.logo_url,
    )


@router.get("/models", response_model=llm_model.LlmModelsResponse)
async def list_models(
    enabled_only: bool = fastapi.Query(
        default=True, description="Only return enabled models"
    ),
):
    """
    List all LLM models available to users.

    Returns models from the in-memory registry cache.
    Use enabled_only=true to filter to only enabled models (default).
    """
    # Get models from in-memory registry
    registry_models = get_enabled_models() if enabled_only else get_all_models()

    # Map to API response models
    models = [
        llm_model.LlmModel(
            slug=model.slug,
            display_name=model.display_name,
            description=model.description,
            provider_name=model.provider_display_name,
            creator=_map_creator(model.creator),
            context_window=model.metadata.context_window,
            max_output_tokens=model.metadata.max_output_tokens,
            price_tier=model.metadata.price_tier,
            is_enabled=model.is_enabled,
            is_recommended=model.is_recommended,
            capabilities=model.capabilities,
            costs=[
                llm_model.LlmModelCost(
                    unit=cost.unit,
                    credit_cost=cost.credit_cost,
                    credential_provider=cost.credential_provider,
                    credential_id=cost.credential_id,
                    credential_type=cost.credential_type,
                    currency=cost.currency,
                    metadata=cost.metadata,
                )
                for cost in model.costs
            ],
        )
        for model in registry_models
    ]

    return llm_model.LlmModelsResponse(models=models, total=len(models))


@router.get("/providers", response_model=llm_model.LlmProvidersResponse)
async def list_providers():
    """
    List all LLM providers with their enabled models.

    Groups enabled models by provider from the in-memory registry.
    """
    # Get all enabled models and group by provider
    registry_models = get_enabled_models()

    # Group models by provider
    provider_map: dict[str, list] = {}
    for model in registry_models:
        provider_key = model.metadata.provider
        if provider_key not in provider_map:
            provider_map[provider_key] = []
        provider_map[provider_key].append(model)

    # Build provider responses
    providers = []
    for provider_key, models in sorted(provider_map.items()):
        # Use the first model's provider display name
        display_name = models[0].provider_display_name if models else provider_key

        providers.append(
            llm_model.LlmProvider(
                name=provider_key,
                display_name=display_name,
                models=[
                    llm_model.LlmModel(
                        slug=model.slug,
                        display_name=model.display_name,
                        description=model.description,
                        provider_name=model.provider_display_name,
                        creator=_map_creator(model.creator),
                        context_window=model.metadata.context_window,
                        max_output_tokens=model.metadata.max_output_tokens,
                        price_tier=model.metadata.price_tier,
                        is_enabled=model.is_enabled,
            is_recommended=model.is_recommended,
                        capabilities=model.capabilities,
                        costs=[
                            llm_model.LlmModelCost(
                                unit=cost.unit,
                                credit_cost=cost.credit_cost,
                                credential_provider=cost.credential_provider,
                                credential_id=cost.credential_id,
                                credential_type=cost.credential_type,
                                currency=cost.currency,
                                metadata=cost.metadata,
                            )
                            for cost in model.costs
                        ],
                    )
                    for model in sorted(models, key=lambda m: m.display_name)
                ],
            )
        )

    return llm_model.LlmProvidersResponse(providers=providers)
