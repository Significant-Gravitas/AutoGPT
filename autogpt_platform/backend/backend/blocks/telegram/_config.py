"""Provider registration for Telegram — metadata only."""

from backend.sdk import ProviderBuilder

telegram = (
    ProviderBuilder("telegram")
    .with_description("Bot messaging and groups")
    .with_supported_auth_types("api_key")
    .build()
)
