"""
Helper utilities for LLM registry integration with block schemas.

This module handles the dynamic injection of discriminator mappings
and model options from the LLM registry into block schemas.
"""

import logging
from typing import Any

from backend.data import llm_registry

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
    Refresh LLM model options and enum values from the registry.

    Updates both 'options' (for frontend dropdown) and 'enum' (Pydantic validation)
    to reflect only currently enabled models.
    """
    fresh_options = llm_registry.get_llm_model_schema_options()
    if not fresh_options:
        return

    enabled_slugs = {opt.get("value") for opt in fresh_options if isinstance(opt, dict)}

    # Update options array
    if "options" in field_schema:
        field_schema["options"] = fresh_options

    # Filter enum to only enabled models
    if "enum" in field_schema and isinstance(field_schema.get("enum"), list):
        old_enum = field_schema["enum"]
        field_schema["enum"] = [val for val in old_enum if val in enabled_slugs]


def refresh_llm_discriminator_mapping(field_schema: dict[str, Any]) -> None:
    """
    Refresh discriminator_mapping for fields that use model-based discrimination.

    The discriminator is already set when AICredentialsField() creates the field.
    We only need to refresh the mapping when models are added/removed.
    """
    if field_schema.get("discriminator") != "model":
        return

    # Always refresh the mapping to get latest models
    fresh_mapping = llm_registry.get_llm_discriminator_mapping()
    if fresh_mapping:
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
