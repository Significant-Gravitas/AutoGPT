"""Provider registration for Smartlead — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

smartlead = (
    ProviderBuilder("smartlead")
    .with_description("Cold email outreach at scale")
    .with_supported_auth_types("api_key")
    .build()
)
