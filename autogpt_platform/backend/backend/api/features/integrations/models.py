"""
Models for integration-related data structures that need to be exposed in the OpenAPI schema.

This module provides models that will be included in the OpenAPI schema generation,
allowing frontend code generators like Orval to create corresponding TypeScript types.
"""

from pydantic import BaseModel, Field

from backend.data.model import CredentialsType
from backend.integrations.providers import ProviderName
from backend.sdk.registry import AutoRegistry
from backend.copilot.subscription_providers import known_profiles


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
    display_alias: str | None = Field(
        default=None,
        description=(
            "Provider slug this one is filed under in the connections UI, "
            "when that differs from ``name``. A ChatGPT subscription is a "
            "different credential from an OpenAI API key but the same entry "
            "to a person looking for it, so ``codex`` is filed under "
            "``openai``. ``None`` means the provider is its own entry."
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


def merge_subscription_summaries(
    rows: list[ProviderMetadata],
) -> list[ProviderMetadata]:
    """Fold each subscription's one-line summary into the entry it files under.

    The connections list shows one card per *display* provider, so the entry
    a subscription aliases into is where a user reads what it offers -- and
    the aliased provider's own registry description is developer-facing
    ("Use your ChatGPT plan with Codex App Server"), not what belongs on a
    card. Composed here rather than in the client, which used to carry the
    whole sentence as a literal keyed on ``codex``.

    Only rows still in ``rows`` contribute, so a provider hidden by the
    entitlement gate does not advertise itself on someone else's card.
    """
    present = {row.name for row in rows}
    summaries: dict[str, list[str]] = {}
    for profile in known_profiles():
        if profile.credential_provider is None or profile.display_alias is None:
            continue
        if not profile.connection_summary:
            continue
        if profile.credential_provider not in present:
            continue
        summaries.setdefault(profile.display_alias, []).append(
            profile.connection_summary
        )

    merged: list[ProviderMetadata] = []
    for row in rows:
        extra = summaries.get(row.name)
        if not extra:
            merged.append(row)
            continue
        base = row.description or f"{row.name} models"
        merged.append(
            row.model_copy(update={"description": " or ".join([base, *extra])})
        )
    return merged


def display_alias_for(name: str) -> str | None:
    """The slug a provider is filed under in the connections UI.

    Only subscription providers alias, and the provider table already says
    which slug each one belongs beside -- the client used to carry that as a
    literal `name === "codex" ? "openai"`, which is a fact about a provider
    living in the one place that cannot be told when it changes.
    """
    for profile in known_profiles():
        if profile.credential_provider == name:
            return profile.display_alias
    return None
