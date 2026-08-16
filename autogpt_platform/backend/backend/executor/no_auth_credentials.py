from typing import Any

from backend.data.model import APIKeyCredentials, CredentialsFieldInfo
from backend.integrations.credentials_store import ollama_credentials, provider_matches


def _can_defer_unresolved_discriminator(field_info: CredentialsFieldInfo) -> bool:
    """Whether an unresolved discriminator can be safely re-checked at runtime.

    Preflight cannot know the discriminator value when it is supplied by an upstream
    link. We may defer the credential decision only when every credential-free
    discriminator variant is backed by the runtime adapter available in this module.
    Authenticated variants are still checked normally once their discriminator value
    is known.
    """
    if not (field_info.discriminator_mapping and field_info.discriminator_type_mapping):
        return False

    credential_free_providers = {
        field_info.discriminator_mapping[value]
        for value, supported_types in field_info.discriminator_type_mapping.items()
        if not supported_types and value in field_info.discriminator_mapping
    }
    if not credential_free_providers:
        return False

    return all(
        provider_matches(str(provider), ollama_credentials.provider)
        for provider in credential_free_providers
    )


def get_no_auth_credentials(
    field_info: CredentialsFieldInfo, input_data: dict[str, Any]
) -> APIKeyCredentials | None:
    """Return the internal runtime credential for credential-free variants.

    CredentialsField remains required at the block schema level so providers that
    require authentication still fail preflight normally. A discriminator variant
    with no supported credential types is therefore an explicit no-auth contract,
    not a blanket optional-credentials flag.

    If the discriminator is unresolved during preflight (for example because the
    model comes from an upstream link), returning the Ollama adapter acts only as a
    defer signal. The executor calls this helper again after linked inputs are
    resolved, so authenticated models still follow the normal credential path.
    """
    if not (field_info.discriminator and field_info.discriminator_mapping):
        return None

    discriminator_value = input_data.get(field_info.discriminator)
    if discriminator_value is None:
        if _can_defer_unresolved_discriminator(field_info):
            return ollama_credentials
        return None

    discriminated_info = field_info.discriminate(discriminator_value)
    if discriminated_info.supported_types:
        return None

    provider = next(iter(discriminated_info.provider), None)
    if provider is not None and provider_matches(
        str(provider), ollama_credentials.provider
    ):
        return ollama_credentials

    raise ValueError(
        f"Credential-free provider {provider!s} has no runtime credential adapter"
    )
