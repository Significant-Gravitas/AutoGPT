from typing import Literal

from pydantic import SecretStr

from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
)
from backend.integrations.providers import ProviderName
from backend.util.settings import Secrets

secrets = Secrets()
HUBSPOT_OAUTH_IS_CONFIGURED = bool(
    secrets.hubspot_client_id and secrets.hubspot_client_secret
)

HubSpotCredentials = APIKeyCredentials | OAuth2Credentials
HubSpotCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.HUBSPOT],
    (
        Literal["api_key", "oauth2"]
        if HUBSPOT_OAUTH_IS_CONFIGURED
        else Literal["api_key"]
    ),
]

CONTACT_SCOPES = {"crm.objects.contacts.read", "crm.objects.contacts.write"}
COMPANY_SCOPES = {"crm.objects.companies.read", "crm.objects.companies.write"}


def HubSpotCredentialsField(scopes: set[str]) -> HubSpotCredentialsInput:
    """Creates a HubSpot credentials input on a block.

    Params:
        scopes: The HubSpot OAuth scopes the block needs. A service key or
            private app token must carry the same scopes.
    """
    return CredentialsField(
        required_scopes=scopes,
        description=(
            "Sign in with HubSpot, or use a HubSpot service key, sent as a Bearer "
            "token. Create one in HubSpot under Development > Keys > Service keys "
            "with the scopes for the objects you use: crm.objects.companies.read "
            "and crm.objects.companies.write for companies, "
            "crm.objects.contacts.read and crm.objects.contacts.write for "
            "contacts and email engagements. An existing private app access "
            "token works too. HubSpot no longer accepts legacy API keys."
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
