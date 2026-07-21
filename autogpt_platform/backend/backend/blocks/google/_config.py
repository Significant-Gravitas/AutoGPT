"""Provider registration for Google — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

google = (
    ProviderBuilder("google")
    .with_description("Gmail, Drive, Calendar, Sheets")
    .with_supported_auth_types("oauth2")
    .build()
)
