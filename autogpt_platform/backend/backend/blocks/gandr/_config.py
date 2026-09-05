"""Provider registration for Gandr (metadata only)."""

from backend.sdk import ProviderBuilder

gandr = (
    ProviderBuilder("gandr")
    .with_description("Text to speech in 23 languages")
    .with_api_key("GANDR_API_KEY", "Gandr API Key")
    .build()
)
