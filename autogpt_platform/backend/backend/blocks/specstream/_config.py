"""Provider registration and shared test credentials for the SpecStream blocks."""

from backend.sdk import APIKeyCredentials, ProviderBuilder, SecretStr

specstream = (
    ProviderBuilder("specstream")
    .with_description(
        "Datasets from the live web, with a source URL and read time on every value read from a page"
    )
    .with_api_key("SPECSTREAM_API_KEY", "SpecStream API Key")
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="specstream",
    api_key=SecretStr("mock-specstream-api-key"),
    title="Mock SpecStream API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
