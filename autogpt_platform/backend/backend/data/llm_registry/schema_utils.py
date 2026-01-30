"""
Helper utilities for LLM registry integration with block schemas.

This module handles the dynamic injection of discriminator mappings
and model options from the LLM registry into block schemas.
"""

import logging
from typing import Any

from backend.data.llm_registry.registry import (
    get_all_model_slugs_for_validation,
    get_default_model_slug,
    get_llm_discriminator_mapping,
    get_llm_model_schema_options,
)

logger = logging.getLogger(__name__)


def is_llm_model_field(field_name: str, field_info: Any) -> bool:
    """
    Check if a field is an LLM model selection field.

    Returns True if the field has 'options' in json_schema_extra
    (set by llm_model_schema_extra() in blocks/llm.py).
    """
    if not hasattr(field_info, "json_schema_extra"):
        return False

    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        return "options" in extra

    return False


def refresh_llm_model_options(field_schema: dict[str, Any]) -> None:
    """
    Refresh LLM model options from the registry.

    Updates 'options' (for frontend dropdown) to show only enabled models,
    but keeps the 'enum' (for validation) inclusive of ALL known models.

    This is important because:
    - Options: What users see in the dropdown (enabled models only)
    - Enum: What values pass validation (all known models, including disabled)

    Existing graphs may have disabled models selected - they should pass validation
    and the fallback logic in llm_call() will handle using an alternative model.
    """
    fresh_options = get_llm_model_schema_options()
    if not fresh_options:
        return

    # Update options array (UI dropdown) - only enabled models
    if "options" in field_schema:
        field_schema["options"] = fresh_options

    all_known_slugs = get_all_model_slugs_for_validation()
    if all_known_slugs and "enum" in field_schema:
        existing_enum = set(field_schema.get("enum", []))
        combined_enum = existing_enum | all_known_slugs
        field_schema["enum"] = sorted(combined_enum)

    # Set the default value from the registry (gpt-4o if available, else first enabled)
    # This ensures new blocks have a sensible default pre-selected
    default_slug = get_default_model_slug()
    if default_slug:
        field_schema["default"] = default_slug


def refresh_llm_discriminator_mapping(field_schema: dict[str, Any]) -> None:
    """
    Refresh discriminator_mapping for fields that use model-based discrimination.

    The discriminator is already set when AICredentialsField() creates the field.
    We only need to refresh the mapping when models are added/removed.
    """
    if field_schema.get("discriminator") != "model":
        return

    # Always refresh the mapping to get latest models
    fresh_mapping = get_llm_discriminator_mapping()
    if fresh_mapping is not None:
        field_schema["discriminator_mapping"] = fresh_mapping


def update_schema_with_llm_registry(
    schema: dict[str, Any], model_class: type | None = None
) -> None:
    """
    Update a JSON schema with current LLM registry data.

    Refreshes:
    1. Model options for LLM model selection fields (dropdown choices)
    2. Discriminator mappings for credentials fields (model → provider)

    Args:
        schema: The JSON schema to update (mutated in-place)
        model_class: The Pydantic model class (optional, for field introspection)
    """
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue

        # Refresh model options for LLM model fields
        if model_class and hasattr(model_class, "model_fields"):
            field_info = model_class.model_fields.get(field_name)
            if field_info and is_llm_model_field(field_name, field_info):
                try:
                    refresh_llm_model_options(field_schema)
                except Exception as exc:
                    logger.warning(
                        "Failed to refresh LLM options for field %s: %s",
                        field_name,
                        exc,
                    )

        # Refresh discriminator mapping for fields that use model discrimination
        try:
            refresh_llm_discriminator_mapping(field_schema)
        except Exception as exc:
            logger.warning(
                "Failed to refresh discriminator mapping for field %s: %s",
                field_name,
                exc,
            )
