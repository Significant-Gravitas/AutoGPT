"""Provider registration for Enrichlayer — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

enrichlayer = (
    ProviderBuilder("enrichlayer")
    .with_description("Enrich leads with company data")
    .with_supported_auth_types("api_key")
    .build()
)
