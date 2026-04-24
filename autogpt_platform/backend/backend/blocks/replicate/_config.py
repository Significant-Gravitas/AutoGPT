"""Provider registration for Replicate — metadata only."""

from backend.sdk import ProviderBuilder

replicate = ProviderBuilder("replicate").with_description(
    "Run and host open-source models"
).build()
