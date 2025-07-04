"""
Shared configuration for example blocks using the new SDK pattern.
"""

from backend.sdk import APIKeyCredentials, BlockCostType, ProviderBuilder, SecretStr

# Configure the example service provider
example_service = (
    ProviderBuilder("example-service")
    .with_api_key("EXAMPLE_SERVICE_API_KEY", "Example Service API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

# Test credentials for example service
EXAMPLE_SERVICE_TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="example-service",
    api_key=SecretStr("mock-example-api-key"),
    title="Mock Example Service API key",
    expires_at=None,
)

EXAMPLE_SERVICE_TEST_CREDENTIALS_INPUT = {
    "provider": EXAMPLE_SERVICE_TEST_CREDENTIALS.provider,
    "id": EXAMPLE_SERVICE_TEST_CREDENTIALS.id,
    "type": EXAMPLE_SERVICE_TEST_CREDENTIALS.type,
    "title": EXAMPLE_SERVICE_TEST_CREDENTIALS.title,
}

# Configure the example webhook provider
example_webhook = (
    ProviderBuilder("examplewebhook")
    .with_api_key("EXAMPLE_WEBHOOK_API_KEY", "Example Webhook API Key")
    .with_base_cost(0, BlockCostType.RUN)  # Webhooks typically don't have run costs
    .build()
)

# Advanced provider configuration
advanced_service = (
    ProviderBuilder("advanced-service")
    .with_api_key("ADVANCED_API_KEY", "Advanced Service API Key")
    .with_base_cost(2, BlockCostType.RUN)
    .build()
)


# Example of a provider with custom API client
class CustomAPIProvider:
    """Example custom API client for demonstration."""

    def __init__(self, credentials):
        self.credentials = credentials

    async def request(self, method: str, endpoint: str, **kwargs):
        # Simulated API request
        return {"status": "ok", "data": kwargs.get("data", {})}


# Configure provider with custom API client
custom_api = (
    ProviderBuilder("custom-api")
    .with_api_key("CUSTOM_API_KEY", "Custom API Key")
    .with_api_client(lambda creds: CustomAPIProvider(creds))
    .with_base_cost(3, BlockCostType.RUN)
    .build()
)
