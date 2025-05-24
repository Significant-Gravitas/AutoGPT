from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

ApolloCredentials = APIKeyCredentials
ApolloCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.APOLLO],
    Literal["api_key"],
]

DEFAULT_CREDENTIAL_ID = "544c62b5-1d0f-4156-8fb4-9525f11656eb"
ENV_VAR = "apollo_api_key"
DEFAULT_TITLE = "Use Credits for Apollo"


def default_credentials() -> APIKeyCredentials | None:
    settings = Settings()
    key = getattr(settings.secrets, ENV_VAR, "")
    if not key:
        return None
    return APIKeyCredentials(
        id=DEFAULT_CREDENTIAL_ID,
        provider=ProviderName.APOLLO.value,
        api_key=SecretStr(key),
        title=DEFAULT_TITLE,
        expires_at=None,
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="apollo",
    api_key=SecretStr("mock-apollo-api-key"),
    title="Mock Apollo API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def ApolloCredentialsField() -> ApolloCredentialsInput:
    """
    Creates a Apollo credentials input on a block.
    """
    return CredentialsField(
        description="The Apollo integration can be used with an API Key.",
    )
