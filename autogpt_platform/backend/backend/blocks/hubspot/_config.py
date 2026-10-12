"""Provider registration for HubSpot — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

from ._auth import HUBSPOT_OAUTH_IS_CONFIGURED

hubspot = (
    ProviderBuilder("hubspot")
    .with_description("CRM, contacts, and deals")
    .with_supported_auth_types(
        "api_key", *(["oauth2"] if HUBSPOT_OAUTH_IS_CONFIGURED else [])
    )
    .build()
)
