"""Provider registration for Compass — metadata only (auth lives elsewhere)."""

from backend.sdk import ProviderBuilder

compass = (
    ProviderBuilder("compass")
    .with_description("Geospatial context for agents")
    .with_supported_auth_types("api_key")
    .build()
)
