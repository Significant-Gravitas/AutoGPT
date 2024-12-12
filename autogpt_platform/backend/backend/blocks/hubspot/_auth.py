from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

HubSpotCredentials = APIKeyCredentials
HubSpotCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.HUBSPOT],
    Literal["api_key"],
]


def HubSpotCredentialsField() -> HubSpotCredentialsInput:
    """Creates a HubSpot credentials input on a block."""
    return CredentialsField(
        description="The HubSpot integration requires an API Key.",
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="hubspot",
    api_key=SecretStr("mock-hubspot-api-key"),
    title="Mock HubSpot API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
