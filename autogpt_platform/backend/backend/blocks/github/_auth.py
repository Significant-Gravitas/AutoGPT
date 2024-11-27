from typing import Literal

from autogpt_libs.supabase_integration_credentials_store.types import (
    APIKeyCredentials,
    OAuth2Credentials,
)
from pydantic import SecretStr

from backend.data.model import CredentialsField, CredentialsMetaInput
from backend.util.settings import Secrets

secrets = Secrets()
GITHUB_OAUTH_IS_CONFIGURED = bool(
    secrets.github_client_id and secrets.github_client_secret
)

GithubCredentials = APIKeyCredentials | OAuth2Credentials
GithubCredentialsInput = CredentialsMetaInput[
    Literal["github"],
    Literal["api_key", "oauth2"] if GITHUB_OAUTH_IS_CONFIGURED else Literal["api_key"],
]


def GithubCredentialsField(scope: str) -> GithubCredentialsInput:
    """
    Creates a GitHub credentials input on a block.

    Params:
        scope: The authorization scope needed for the block to work. ([list of available scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes))
    """  # noqa
    return CredentialsField(
        provider="github",
        supported_credential_types=(
            {"api_key", "oauth2"} if GITHUB_OAUTH_IS_CONFIGURED else {"api_key"}
        ),
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
