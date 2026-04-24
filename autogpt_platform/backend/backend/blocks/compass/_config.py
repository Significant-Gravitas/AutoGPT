"""Provider registration for Compass — metadata only (auth lives elsewhere)."""

from backend.sdk import ProviderBuilder

compass = ProviderBuilder("compass").with_description(
    "Geospatial context for agents"
).build()
