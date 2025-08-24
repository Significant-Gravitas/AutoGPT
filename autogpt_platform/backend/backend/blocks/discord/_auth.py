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
DISCORD_OAUTH_IS_CONFIGURED = bool(
    secrets.discord_client_id and secrets.discord_client_secret
)

# Bot token credentials (existing)
DiscordBotCredentials = APIKeyCredentials
DiscordBotCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DISCORD], Literal["api_key"]
]

# OAuth2 credentials (new)
DiscordOAuthCredentials = OAuth2Credentials
DiscordOAuthCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DISCORD], Literal["oauth2"]
]


def DiscordBotCredentialsField() -> DiscordBotCredentialsInput:
    """Creates a Discord bot token credentials field."""
    return CredentialsField(description="Discord bot token")


def DiscordOAuthCredentialsField(scopes: list[str]) -> DiscordOAuthCredentialsInput:
    """Creates a Discord OAuth2 credentials field."""
    return CredentialsField(
        description="Discord OAuth2 credentials",
        required_scopes=set(scopes) | {"identify"},  # Basic user info scope
    )


# Test credentials for bot tokens
TEST_BOT_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="discord",
    api_key=SecretStr("test_api_key"),
    title="Mock Discord API key",
    expires_at=None,
)
TEST_BOT_CREDENTIALS_INPUT = {
    "provider": TEST_BOT_CREDENTIALS.provider,
    "id": TEST_BOT_CREDENTIALS.id,
    "type": TEST_BOT_CREDENTIALS.type,
    "title": TEST_BOT_CREDENTIALS.type,
}

# Test credentials for OAuth2
TEST_OAUTH_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="discord",
    access_token=SecretStr("test_access_token"),
    title="Mock Discord OAuth",
    scopes=["identify"],
    username="testuser",
)
TEST_OAUTH_CREDENTIALS_INPUT = {
    "provider": TEST_OAUTH_CREDENTIALS.provider,
    "id": TEST_OAUTH_CREDENTIALS.id,
    "type": TEST_OAUTH_CREDENTIALS.type,
    "title": TEST_OAUTH_CREDENTIALS.type,
}
