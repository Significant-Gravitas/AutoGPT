"""                                                                                       
Slack Bot credentials handling.                                                           
                  
Slack bots use a Bot Token (xoxb-...) obtained from the Slack API dashboard.
"""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

SlackCredentials = APIKeyCredentials
SlackCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SLACK], Literal["api_key"]
]


def SlackCredentialsField() -> SlackCredentialsInput:
    """Creates a Slack bot token credentials field."""
    return CredentialsField(
        description="Slack Bot Token (xoxb-...). "
        "Create a Slack app at https://api.slack.com/apps, add the required "
        "OAuth scopes, and install it to your workspace to get the token."
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="slack",
    api_key=SecretStr("mock-slack-bot-token"),
    title="Mock Slack Bot Token",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
