from tenki import AsyncClient

from backend.sdk import APIKeyCredentials, BlockCostType, ProviderBuilder, SecretStr

tenki = (
    ProviderBuilder("tenki")
    .with_description("Ephemeral cloud sandboxes for secure code execution")
    .with_api_key("TENKI_API_KEY", "Tenki API Key")
    .with_base_cost(1, BlockCostType.SECOND, cost_divisor=10)
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="tenki",
    title="Mock Tenki API Key",
    api_key=SecretStr("mock-tenki-api-key"),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "id": TEST_CREDENTIALS.id,
    "provider": TEST_CREDENTIALS.provider,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def create_client(credentials: APIKeyCredentials) -> AsyncClient:
    return AsyncClient(auth_token=credentials.api_key.get_secret_value())
