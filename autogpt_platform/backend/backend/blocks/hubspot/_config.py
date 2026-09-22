"""Provider registration for HubSpot — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

hubspot = (
    ProviderBuilder("hubspot")
    .with_description("CRM, contacts, and deals")
    .with_supported_auth_types("api_key", "oauth2")
    .build()
)
