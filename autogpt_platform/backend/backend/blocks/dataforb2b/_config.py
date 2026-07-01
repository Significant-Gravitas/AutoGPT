"""Provider registration for DataForB2B.

Registers the provider description shown in the settings integrations UI.
DataForB2B uses API-key auth (header ``api_key``), set up in ``_auth.py``.
"""

from backend.sdk import ProviderBuilder

dataforb2b = (
    ProviderBuilder("dataforb2b")
    .with_description(
        "B2B data API — search leads, enrich profiles (from a LinkedIn URL), find work "
        "emails and phone numbers for prospecting and outreach."
    )
    .with_supported_auth_types("api_key")
    .build()
)
