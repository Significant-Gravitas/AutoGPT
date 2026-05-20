"""
Models for integration-related data structures that need to be exposed in the OpenAPI schema.

This module provides models that will be included in the OpenAPI schema generation,
allowing frontend code generators like Orval to create corresponding TypeScript types.
"""

from pydantic import BaseModel, Field

from backend.data.model import CredentialsType
from backend.integrations.providers import ProviderName
from backend.sdk.registry import AutoRegistry


def get_all_provider_names() -> list[str]:
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

    providers: list[str] = Field(
        description="List of all available provider names",
        default_factory=get_all_provider_names,
    )


class ProviderMetadata(BaseModel):
    """Display metadata for a provider, shown in the settings integrations UI."""

    name: str = Field(description="Provider slug (e.g. ``github``)")
    description: str | None = Field(
        default=None,
        description=(
            "One-line human-readable summary of what the provider does. "
            "Declared via ``ProviderBuilder.with_description(...)`` in the "
            "provider's ``_config.py``. ``None`` if not set."
        ),
    )
    supported_auth_types: list[CredentialsType] = Field(
        default_factory=list,
        description=(
            "Credential types this provider accepts. Drives which connection "
            "tabs the settings UI renders for the provider. Empty list means "
            "no auth types declared."
        ),
    )


def get_supported_auth_types(name: str) -> list[CredentialsType]:
    """Return the provider's supported credential types from :class:`AutoRegistry`.

    Populated by :meth:`ProviderBuilder.with_supported_auth_types` (or by
    ``with_oauth`` / ``with_api_key`` / ``with_user_password`` when the provider
    uses the full builder chain). Returns an empty list for providers with no
    auth types declared.
    """
    provider = AutoRegistry.get_provider(name)
    if provider is None:
        return []
    return sorted(provider.supported_auth_types)


def get_provider_description(name: str) -> str | None:
    """Return the provider's description from :class:`AutoRegistry`.

    Descriptions are declared via ``ProviderBuilder.with_description(...)`` in
    the provider's ``_config.py`` (SDK path) or in
    ``blocks/_static_provider_configs.py`` (for providers that don't yet have
    their own directory). Returns ``None`` for providers with no registered
    description.
    """
    provider = AutoRegistry.get_provider(name)
    if provider is None:
        return None
    return provider.description


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
        examples=[
            {
                "OPENAI": "openai",
                "ANTHROPIC": "anthropic",
                "EXA": "exa",
                "GEM": "gem",
                "EXAMPLE_SERVICE": "example-service",
            }
        ],
    )
