"""Provider registration for ElevenLabs — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

elevenlabs = (
    ProviderBuilder("elevenlabs")
    .with_description("Realistic AI voice synthesis")
    .with_supported_auth_types("api_key")
    .build()
)
