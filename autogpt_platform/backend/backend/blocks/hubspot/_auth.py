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
        description=(
            "A HubSpot service key or private app access token, sent as a Bearer "
            "token. Create a service key in HubSpot under Development > Keys > "
            "Service keys, or a private app under Development > Legacy apps, with "
            "the scopes for the objects you use: crm.objects.companies.read and "
            "crm.objects.companies.write for companies, crm.objects.contacts.read "
            "and crm.objects.contacts.write for contacts and email engagements. "
            "HubSpot no longer accepts legacy API keys."
        ),
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="hubspot",
    api_key=SecretStr("mock-hubspot-access-token"),
    title="Mock HubSpot access token",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
