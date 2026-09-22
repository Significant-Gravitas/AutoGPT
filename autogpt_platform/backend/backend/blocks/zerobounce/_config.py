"""Provider registration for ZeroBounce — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

zerobounce = (
    ProviderBuilder("zerobounce")
    .with_description("Email address verification")
    .with_supported_auth_types("api_key")
    .build()
)
