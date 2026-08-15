from typing import Any

from backend.data.model import APIKeyCredentials, CredentialsFieldInfo
from backend.integrations.credentials_store import ollama_credentials, provider_matches


def get_no_auth_credentials(
    field_info: CredentialsFieldInfo, input_data: dict[str, Any]
) -> APIKeyCredentials | None:
    """Return an internal runtime credential for credential-free variants.

    CredentialsField remains required at the block schema level so providers that
    require authentication still fail preflight normally. A discriminator variant
    with no supported credential types is therefore an explicit no-auth contract,
    not a blanket optional-credentials flag.
    """
    if not (field_info.discriminator and field_info.discriminator_mapping):
        return None

    discriminator_value = input_data.get(field_info.discriminator)
    if discriminator_value is None:
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
