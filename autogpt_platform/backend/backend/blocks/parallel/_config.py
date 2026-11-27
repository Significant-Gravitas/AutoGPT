"""Shared provider and credential configuration for Parallel blocks."""

from backend.sdk import APIKeyCredentials, BlockCostType, ProviderBuilder, SecretStr

# Parallel bills the Decisions API at $1 per 1,000 requests ($0.001 each),
# whatever the number of questions in the request. 1 credit per run is the
# smallest whole charge that covers it; the exact USD cost is also recorded
# per run as provider_cost.
parallel = (
    ProviderBuilder("parallel")
    .with_description(
        "Web-grounded decisions: typed yes/no, choice and score answers "
        "backed by current web sources"
    )
    .with_api_key("PARALLEL_API_KEY", "Parallel API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="7c1f6a52-3c8e-4d0b-9a3e-5b2f0d9e8c41",
    provider="parallel",
    api_key=SecretStr("mock-parallel-api-key"),
    title="Mock Parallel API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
