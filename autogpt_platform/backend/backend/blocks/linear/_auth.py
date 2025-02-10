from enum import Enum
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
LINEAR_OAUTH_IS_CONFIGURED = bool(
    secrets.linear_client_id and secrets.linear_client_secret
)

LinearCredentials = OAuth2Credentials | APIKeyCredentials
# LinearCredentialsInput = CredentialsMetaInput[
#     Literal[ProviderName.LINEAR],
#     Literal["oauth2", "api_key"] if LINEAR_OAUTH_IS_CONFIGURED else Literal["oauth2"],
# ]
LinearCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.LINEAR], Literal["oauth2"]
]


# (required) Comma separated list of scopes:

# read - (Default) Read access for the user's account. This scope will always be present.

# write - Write access for the user's account. If your application only needs to create comments, use a more targeted scope

# issues:create - Allows creating new issues and their attachments

# comments:create - Allows creating new issue comments

# timeSchedule:write - Allows creating and modifying time schedules


# admin - Full access to admin level endpoints. You should never ask for this permission unless it's absolutely needed
class LinearScope(str, Enum):
    READ = "read"
    WRITE = "write"
    ISSUES_CREATE = "issues:create"
    COMMENTS_CREATE = "comments:create"
    TIME_SCHEDULE_WRITE = "timeSchedule:write"
    ADMIN = "admin"


def LinearCredentialsField(scopes: list[LinearScope]) -> LinearCredentialsInput:
    """
    Creates a Linear credentials input on a block.

    Params:
        scope: The authorization scope needed for the block to work. ([list of available scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes))
    """  # noqa
    return CredentialsField(
        required_scopes=set([LinearScope.READ.value]).union(
            set([scope.value for scope in scopes])
        ),
        description="The Linear integration can be used with OAuth, "
        "or any API key with sufficient permissions for the blocks it is used on.",
    )


TEST_CREDENTIALS_OAUTH = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="linear",
    title="Mock Linear API key",
    username="mock-linear-username",
    access_token=SecretStr("mock-linear-access-token"),
    access_token_expires_at=None,
    refresh_token=SecretStr("mock-linear-refresh-token"),
    refresh_token_expires_at=None,
    scopes=["mock-linear-scopes"],
)

TEST_CREDENTIALS_API_KEY = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="linear",
    title="Mock Linear API key",
    api_key=SecretStr("mock-linear-api-key"),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT_OAUTH = {
    "provider": TEST_CREDENTIALS_OAUTH.provider,
    "id": TEST_CREDENTIALS_OAUTH.id,
    "type": TEST_CREDENTIALS_OAUTH.type,
    "title": TEST_CREDENTIALS_OAUTH.type,
}

TEST_CREDENTIALS_INPUT_API_KEY = {
    "provider": TEST_CREDENTIALS_API_KEY.provider,
    "id": TEST_CREDENTIALS_API_KEY.id,
    "type": TEST_CREDENTIALS_API_KEY.type,
    "title": TEST_CREDENTIALS_API_KEY.type,
}
