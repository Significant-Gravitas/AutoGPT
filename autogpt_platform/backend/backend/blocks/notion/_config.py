"""Provider registration for Notion — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

notion = (
    ProviderBuilder("notion")
    .with_description("Pages, databases, and blocks")
    .with_supported_auth_types("oauth2")
    .build()
)
