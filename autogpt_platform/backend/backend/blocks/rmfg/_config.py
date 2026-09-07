"""Provider registration and shared test credentials for the RMFG blocks."""

from backend.sdk import APIKeyCredentials, ProviderBuilder, SecretStr

from ._webhook import RMFGWebhooksManager

rmfg = (
    ProviderBuilder("rmfg")
    .with_description("Sheet-metal and tube-laser manufacturing quotes and orders")
    .with_api_key("RMFG_API_KEY", "RMFG API Key")
    .with_webhook_manager(RMFGWebhooksManager)
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="rmfg",
    api_key=SecretStr("mock-rmfg-api-key"),
    title="Mock RMFG API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
