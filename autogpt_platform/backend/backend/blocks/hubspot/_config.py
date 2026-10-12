"""Provider registration for HubSpot — metadata only (auth lives in ``_auth.py``)."""

from backend.data.model import CredentialsType
from backend.sdk import ProviderBuilder

from ._auth import HUBSPOT_OAUTH_IS_CONFIGURED

# OAuth is offered only once the HubSpot app is configured; until then a sign-in
# button would only lead to a "not configured" error.
auth_types: list[CredentialsType] = ["api_key"]
if HUBSPOT_OAUTH_IS_CONFIGURED:
    auth_types.append("oauth2")

hubspot = (
    ProviderBuilder("hubspot")
    .with_description("CRM, contacts, and deals")
    .with_supported_auth_types(*auth_types)
    .build()
)
