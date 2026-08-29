from typing import Literal

from pydantic import SecretStr

from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    ProviderName,
)
from backend.integrations.oauth.todoist import TodoistOAuthHandler
from backend.util.settings import Secrets

secrets = Secrets()
TODOIST_OAUTH_IS_CONFIGURED = bool(
    secrets.todoist_client_id and secrets.todoist_client_secret
)

TodoistCredentials = OAuth2Credentials
TodoistCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.TODOIST], Literal["oauth2"]
]


def TodoistCredentialsField(scopes: list[str]) -> TodoistCredentialsInput:
    """
    Creates a Todoist credentials input on a block.

    Params:
        scopes: The authorization scopes needed for the block to work.
    """
    return CredentialsField(
        required_scopes=set(TodoistOAuthHandler.DEFAULT_SCOPES + scopes),
        description="The Todoist integration requires OAuth2 authentication.",
    )


TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="todoist",
    access_token=SecretStr("mock-todoist-access-token"),
    refresh_token=None,
    access_token_expires_at=None,
    scopes=[
        "task:add",
        "data:read",
        "data:read_write",
        "data:delete",
        "project:delete",
    ],
    title="Mock Todoist OAuth2 Credentials",
    username="mock-todoist-username",
    refresh_token_expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
