"""Provider registration for Todoist — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

todoist = (
    ProviderBuilder("todoist")
    .with_description("Tasks and projects")
    .with_supported_auth_types("oauth2")
    .build()
)
