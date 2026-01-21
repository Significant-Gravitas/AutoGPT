"""
LLM Registry module for managing LLM models, providers, and costs dynamically.

This module provides a database-driven registry system for LLM models,
replacing hardcoded model configurations with a flexible admin-managed system.
"""

from backend.data.llm_registry.model import ModelMetadata

# Re-export for backwards compatibility
from backend.data.llm_registry.notifications import (
    REGISTRY_REFRESH_CHANNEL,
    publish_registry_refresh_notification,
    subscribe_to_registry_refresh,
)
from backend.data.llm_registry.registry import (
    RegistryModel,
    RegistryModelCost,
    RegistryModelCreator,
    get_all_model_slugs_for_validation,
    get_default_model_slug,
    get_dynamic_model_slugs,
    get_fallback_model_for_disabled,
    get_llm_discriminator_mapping,
    get_llm_model_cost,
    get_llm_model_metadata,
    get_llm_model_schema_options,
    get_model_info,
    is_model_enabled,
    iter_dynamic_models,
    refresh_llm_registry,
    register_static_costs,
    register_static_metadata,
)
from backend.data.llm_registry.schema_utils import (
    is_llm_model_field,
    refresh_llm_discriminator_mapping,
    refresh_llm_model_options,
    update_schema_with_llm_registry,
)

__all__ = [
    # Types
    "ModelMetadata",
    "RegistryModel",
    "RegistryModelCost",
    "RegistryModelCreator",
    # Registry functions
    "get_all_model_slugs_for_validation",
    "get_default_model_slug",
    "get_dynamic_model_slugs",
    "get_fallback_model_for_disabled",
    "get_llm_discriminator_mapping",
    "get_llm_model_cost",
    "get_llm_model_metadata",
    "get_llm_model_schema_options",
    "get_model_info",
    "is_model_enabled",
    "iter_dynamic_models",
    "refresh_llm_registry",
    "register_static_costs",
    "register_static_metadata",
    # Notifications
    "REGISTRY_REFRESH_CHANNEL",
    "publish_registry_refresh_notification",
    "subscribe_to_registry_refresh",
    # Schema utilities
    "is_llm_model_field",
    "refresh_llm_discriminator_mapping",
    "refresh_llm_model_options",
    "update_schema_with_llm_registry",
]
