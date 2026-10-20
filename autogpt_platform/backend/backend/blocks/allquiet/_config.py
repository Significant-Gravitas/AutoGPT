"""Shared configuration for all All Quiet blocks."""

from backend.sdk import APIKeyCredentials, ProviderBuilder, SchemaField, SecretStr

from ._types import AllQuietRegion
from ._webhook import AllQuietWebhooksManager

# All Quiet bills the Public API as part of the flat PRO/ENTERPRISE subscription
# rather than per request, so there is no base cost to pass through here.
allquiet = (
    ProviderBuilder("allquiet")
    .with_description("Incident management and on-call scheduling")
    .with_api_key("ALLQUIET_API_KEY", "All Quiet API Key")
    .with_webhook_manager(AllQuietWebhooksManager)
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="allquiet",
    api_key=SecretStr("mock-allquiet-api-key"),
    title="Mock All Quiet API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def region_field() -> AllQuietRegion:
    """Shared input field: which All Quiet deployment to talk to."""
    return SchemaField(
        title="Region",
        description=(
            "The All Quiet deployment your API key belongs to. "
            "Use EU if you signed up on allquiet.eu."
        ),
        default=AllQuietRegion.US,
        advanced=True,
    )
