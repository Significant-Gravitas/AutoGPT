"""Provider registration for Smartlead — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

smartlead = ProviderBuilder("smartlead").with_description(
    "Cold email outreach at scale"
).build()
