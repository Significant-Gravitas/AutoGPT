"""
Shared configuration for all Slack blocks using the SDK pattern.
"""

from enum import Enum

from backend.sdk import APIKeyCredentials, OAuth2Credentials, ProviderBuilder, SecretStr

from ._oauth import SlackOAuthHandler


class SlackScope(str, Enum):
    """Slack bot OAuth scopes.

    chat:write - (Required) Send messages as the app's bot user.
    chat:write.public - Post to public channels without an explicit /invite.
    chat:write.customize - Override the bot's username/icon per message.
    """

    CHAT_WRITE = "chat:write"
    CHAT_WRITE_PUBLIC = "chat:write.public"
    CHAT_WRITE_CUSTOMIZE = "chat:write.customize"


slack = (
    ProviderBuilder("slack")
    .with_description("Send messages to Slack channels and DMs")
    # Deliberately not .with_api_key(env_var_name=...): that would auto-create
    # a shared default credential from an org-wide env var and expose one
    # Slack workspace's bot token to every user of the platform. Each user
    # pastes their own bot token instead.
    .with_supported_auth_types("api_key")
    .with_oauth(
        SlackOAuthHandler,
        scopes=[
            SlackScope.CHAT_WRITE.value,
            SlackScope.CHAT_WRITE_PUBLIC.value,
            SlackScope.CHAT_WRITE_CUSTOMIZE.value,
        ],
        client_id_env_var="SLACK_CLIENT_ID",
        client_secret_env_var="SLACK_CLIENT_SECRET",
    )
    .build()
)


TEST_CREDENTIALS_API_KEY = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="slack",
    api_key=SecretStr("mock-slack-bot-token"),
    title="Mock Slack Bot Token",
    expires_at=None,
)

TEST_CREDENTIALS_OAUTH = OAuth2Credentials(
    id="12345678-9abc-def0-1234-56789abcdef0",
    provider="slack",
    title="Mock Slack OAuth",
    username="mock-slack-workspace",
    access_token=SecretStr("mock-slack-oauth-access-token"),
    access_token_expires_at=None,
    refresh_token=None,
    refresh_token_expires_at=None,
    scopes=[SlackScope.CHAT_WRITE.value],
)

TEST_CREDENTIALS_INPUT_API_KEY = {
    "provider": TEST_CREDENTIALS_API_KEY.provider,
    "id": TEST_CREDENTIALS_API_KEY.id,
    "type": TEST_CREDENTIALS_API_KEY.type,
    "title": TEST_CREDENTIALS_API_KEY.title,
}

TEST_CREDENTIALS_INPUT_OAUTH = {
    "provider": TEST_CREDENTIALS_OAUTH.provider,
    "id": TEST_CREDENTIALS_OAUTH.id,
    "type": TEST_CREDENTIALS_OAUTH.type,
    "title": TEST_CREDENTIALS_OAUTH.title,
}
