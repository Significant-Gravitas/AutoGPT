"""Provider registration for Discord — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

discord = (
    ProviderBuilder("discord")
    .with_description("Messages, channels, and servers")
    .with_supported_auth_types("api_key", "oauth2")
    .build()
)
