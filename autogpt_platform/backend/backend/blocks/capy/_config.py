"""Shared provider and credential configuration for the Capy blocks."""

from backend.sdk import APIKeyCredentials, ProviderBuilder, SecretStr

# Users bring their own Capy key (Settings → API in the Capy app). Capy bills
# the work to that key's organization, so there is no platform-held system key
# and no base cost to pass through here.
capy = (
    ProviderBuilder("capy")
    .with_description(
        "Cloud coding agents: start and steer agent threads on your repos, "
        "and run pull request reviews"
    )
    .with_supported_auth_types("api_key")
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="3c9a4b0e-5d1f-4e27-9a8b-6f0c2d1e7a54",
    provider="capy",
    api_key=SecretStr("capy_mock-api-key"),
    title="Mock Capy API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def capy_credentials_field():
    return capy.credentials_field(
        description=(
            "A Capy API key, minted in the Capy app under Settings → API. "
            "Threads it creates belong to the key's principal."
        )
    )
