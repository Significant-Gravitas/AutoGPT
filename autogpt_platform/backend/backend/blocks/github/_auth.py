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
GITHUB_OAUTH_IS_CONFIGURED = bool(
    secrets.github_client_id and secrets.github_client_secret
)

GithubCredentials = APIKeyCredentials | OAuth2Credentials
GithubCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.GITHUB],
    Literal["api_key", "oauth2"] if GITHUB_OAUTH_IS_CONFIGURED else Literal["api_key"],
]

GithubFineGrainedAPICredentials = APIKeyCredentials
GithubFineGrainedAPICredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.GITHUB], Literal["api_key"]
]


def GithubCredentialsField(scope: str) -> GithubCredentialsInput:
    """
    Creates a GitHub credentials input on a block.

    Params:
        scope: The authorization scope needed for the block to work. ([list of available scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes))
    """  # noqa
    return CredentialsField(
        required_scopes={scope},
        description="The GitHub integration can be used with OAuth, "
        "or any API key with sufficient permissions for the blocks it is used on.",
    )


def GithubFineGrainedAPICredentialsField(
    scope: str,
) -> GithubFineGrainedAPICredentialsInput:
    return CredentialsField(
        required_scopes={scope},
        description="The GitHub integration can be used with OAuth, "
        "or any API key with sufficient permissions for the blocks it is used on.",
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="github",
    api_key=SecretStr("mock-github-api-key"),
    title="Mock GitHub API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}

TEST_FINE_GRAINED_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="github",
    api_key=SecretStr("mock-github-api-key"),
    title="Mock GitHub API key",
    expires_at=None,
)

TEST_FINE_GRAINED_CREDENTIALS_INPUT = {
    "provider": TEST_FINE_GRAINED_CREDENTIALS.provider,
    "id": TEST_FINE_GRAINED_CREDENTIALS.id,
    "type": TEST_FINE_GRAINED_CREDENTIALS.type,
    "title": TEST_FINE_GRAINED_CREDENTIALS.type,
}
