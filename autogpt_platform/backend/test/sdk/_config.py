"""
Shared configuration for SDK test providers using the SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

# Configure test providers
test_api = (
    ProviderBuilder("test_api")
    .with_api_key("TEST_API_KEY", "Test API Key")
    .with_base_cost(5, BlockCostType.RUN)
    .build()
)

test_service = (
    ProviderBuilder("test_service")
    .with_api_key("TEST_SERVICE_API_KEY", "Test Service API Key")
    .with_base_cost(10, BlockCostType.RUN)
    .build()
)
