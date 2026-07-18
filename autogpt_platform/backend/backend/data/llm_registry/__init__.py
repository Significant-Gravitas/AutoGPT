"""LLM Registry - Dynamic model management system."""

from .notifications import (
    publish_registry_refresh_notification,
    subscribe_to_registry_refresh,
)
from .registry import (
    RegistryModel,
    RegistryModelCost,
    RegistryModelCreator,
    RegistryModelMetadata,
    clear_registry_cache,
    get_all_model_slugs_for_validation,
    get_all_models,
    get_default_model_slug,
    get_enabled_models,
    get_model,
    get_route,
    get_schema_options,
    refresh_llm_registry,
)

__all__ = [
    # Models
    "RegistryModel",
    "RegistryModelCost",
    "RegistryModelCreator",
    "RegistryModelMetadata",
    # Cache management
    "clear_registry_cache",
    "publish_registry_refresh_notification",
    "subscribe_to_registry_refresh",
    # Read functions
    "refresh_llm_registry",
    "get_model",
    "get_all_models",
    "get_enabled_models",
    "get_route",
    "get_schema_options",
    "get_default_model_slug",
    "get_all_model_slugs_for_validation",
]
