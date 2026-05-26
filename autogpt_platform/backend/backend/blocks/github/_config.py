"""Provider registration for GitHub — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

github = (
    ProviderBuilder("github")
    .with_description("Issues, pull requests, repositories")
    .with_supported_auth_types("api_key", "oauth2")
    .build()
)
