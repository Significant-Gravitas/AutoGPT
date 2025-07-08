"""
Models for integration-related data structures that need to be exposed in the OpenAPI schema.

This module provides models that will be included in the OpenAPI schema generation,
allowing frontend code generators like Orval to create corresponding TypeScript types.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, create_model

from backend.integrations.providers import ProviderName
from backend.sdk.registry import AutoRegistry


def get_all_provider_names() -> List[str]:
    """
    Collect all provider names from both ProviderName enum and AutoRegistry.

    This function should be called at runtime to ensure we get all
    dynamically registered providers.

    Returns:
        A sorted list of unique provider names.
    """
    # Get static providers from enum
    static_providers = [member.value for member in ProviderName]

    # Get dynamic providers from registry
    dynamic_providers = AutoRegistry.get_all_provider_names()

    # Combine and deduplicate
    all_providers = list(set(static_providers + dynamic_providers))
    all_providers.sort()

    return all_providers


# Note: We don't create a static enum here because providers are registered dynamically.
# Instead, we expose provider names through API endpoints that can be fetched at runtime.


class ProviderNamesResponse(BaseModel):
    """Response containing list of all provider names."""

    providers: List[str] = Field(
        description="List of all available provider names",
        default_factory=get_all_provider_names,
    )


def create_provider_enum_model():
    """
    Dynamically create a model with all provider names as a Literal type.
    This ensures the OpenAPI schema includes all provider names.
    """
    all_providers = get_all_provider_names()

    if not all_providers:
        # Fallback if no providers are registered yet
        all_providers = ["unknown"]

    # Create a Literal type with all provider names
    # This will be included in the OpenAPI schema
    ProviderNameLiteral = Literal[tuple(all_providers)]  # type: ignore

    # Create a dynamic model that uses this Literal
    DynamicProviderModel = create_model(
        "AllProviderNames",
        provider=(
            ProviderNameLiteral,
            Field(description="A provider name from the complete list"),
        ),
        __module__=__name__,
    )

    return DynamicProviderModel


class ProviderConstants(BaseModel):
    """
    Model that exposes all provider names as a constant in the OpenAPI schema.
    This is designed to be converted by Orval into a TypeScript constant.
    """

    PROVIDER_NAMES: dict[str, str] = Field(
        description="All available provider names as a constant mapping",
        default_factory=lambda: {
            name.upper().replace("-", "_"): name for name in get_all_provider_names()
        },
    )

    class Config:
        schema_extra = {
            "example": {
                "PROVIDER_NAMES": {
                    "OPENAI": "openai",
                    "ANTHROPIC": "anthropic",
                    "EXA": "exa",
                    "GEM": "gem",
                    "EXAMPLE_SERVICE": "example-service",
                }
            }
        }
