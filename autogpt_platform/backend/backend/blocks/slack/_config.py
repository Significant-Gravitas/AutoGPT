"""Provider registration for Slack."""

from backend.sdk import ProviderBuilder

slack = (
    ProviderBuilder("slack")
    .with_description("Send messages to channels, DMs, and threads")
    .with_supported_auth_types("api_key")
    .build()
)
