"""Provider registration for WaveSpeed — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

wavespeed = (
    ProviderBuilder("wavespeed")
    .with_description("Fast hosted image and video model inference")
    .with_supported_auth_types("api_key")
    .build()
)
