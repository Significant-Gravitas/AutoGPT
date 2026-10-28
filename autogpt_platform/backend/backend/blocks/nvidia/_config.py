"""Provider registration for Nvidia — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

nvidia = (
    ProviderBuilder("nvidia")
    .with_description("NIM-hosted foundation models")
    .with_supported_auth_types("api_key")
    .build()
)
