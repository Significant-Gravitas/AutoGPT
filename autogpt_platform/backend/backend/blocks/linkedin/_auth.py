from typing import Literal

from pydantic import SecretStr

from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    ProviderName,
)
from backend.integrations.oauth.linkedin import LinkedInOAuthHandler
from backend.util.settings import Secrets

secrets = Secrets()
LINKEDIN_OAUTH_IS_CONFIGURED = bool(
    secrets.linkedin_client_id and secrets.linkedin_client_secret
)

LinkedInCredentials = OAuth2Credentials
LinkedInCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.LINKEDIN], Literal["oauth2"]
]


def LinkedInCredentialsField(scopes: list[str]) -> LinkedInCredentialsInput:
    return CredentialsField(
        required_scopes=set(LinkedInOAuthHandler.DEFAULT_SCOPES + scopes),
        description="Connect your LinkedIn account via OAuth2.",
    )


TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="linkedin",
    access_token=SecretStr("mock-linkedin-access-token"),
    refresh_token=SecretStr("mock-linkedin-refresh-token"),
    access_token_expires_at=9999999999,
    scopes=["openid", "profile", "email", "r_1st_connections_size"],
    title="Mock LinkedIn OAuth2 Credentials",
    username="mock-linkedin-user@example.com",
    refresh_token_expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
