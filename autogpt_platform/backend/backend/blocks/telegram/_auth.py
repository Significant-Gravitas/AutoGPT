"""
Telegram Bot credentials handling.

Telegram bots use an API key (bot token) obtained from @BotFather.
"""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

# Bot token credentials (API key style)
TelegramCredentials = APIKeyCredentials
TelegramCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.TELEGRAM], Literal["api_key"]
]


def TelegramCredentialsField() -> TelegramCredentialsInput:
    """Creates a Telegram bot token credentials field."""
    return CredentialsField(
        description="Telegram Bot API token from @BotFather. "
        "Create a bot at https://t.me/BotFather to get your token."
    )


# Test credentials for unit tests
TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="telegram",
    api_key=SecretStr("test_telegram_bot_token"),
    title="Mock Telegram Bot Token",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
