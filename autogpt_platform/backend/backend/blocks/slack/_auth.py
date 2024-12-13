from enum import Enum
from typing import Literal
from backend.data.model import OAuth2Credentials,ProviderName,CredentialsField, CredentialsMetaInput
from pydantic import SecretStr
from backend.util.settings import Secrets

secrets = Secrets()

SlackProviderName = Literal[ProviderName.SLACK_BOT, ProviderName.SLACK_USER]
SlackCredentials = OAuth2Credentials
SlackCredentialsInput = CredentialsMetaInput[SlackProviderName, Literal["oauth2"]]


def SlackCredentialsField(scopes: list[str]) -> SlackCredentialsInput:
    return CredentialsField(
        description="The Slack integration requires OAuth2 authentication.",
        discriminator="type",
        discriminator_mapping={
            model.value: model.value for model in SlackModel
        },
    )

class SlackModel(str, Enum):
    SLACK_BOT = "slack_bot"
    SLACK_USER = "slack_user"


TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="slack_bot",
    type="oauth2",
    access_token=SecretStr("mock-slack-access-token"),
    refresh_token=SecretStr("mock-slack-refresh-token"),
    access_token_expires_at=1234567890,
    scopes=["chat:write", "channels:read", "users:read", "channels:history"],
    title="Mock Slack OAuth2 Credentials",
    username="mock-slack-username",
    refresh_token_expires_at=1234567890,
)

TEST_CREDENTIALS_INPUT = {
    "provider": "slack_bot",
    "id": TEST_CREDENTIALS.id,
    "type": "oauth2",
    "title": TEST_CREDENTIALS.title,
}
