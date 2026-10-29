"""Shared provider and credential configuration for Treg blocks."""

from backend.integrations.credentials_store import is_system_credential
from backend.sdk import APIKeyCredentials, BlockCostType, ProviderBuilder, SecretStr

# Treg bills each call from the team's prepaid balance at the provider's own
# price (0% markup) and reports the charge in X-Treg-Cost-Micro. Every block
# records that charge as provider_cost, and COST_USD bills platform-key runs
# from it at 150 credits per USD (the 1.5x margin used by other COST_USD
# blocks). Runs on a user's own Treg key are not billed by the platform.
treg = (
    ProviderBuilder("treg")
    .with_description(
        "People, company and web data from 4,000+ pay-per-call tools, "
        "with no provider signup"
    )
    .with_api_key("TREG_API_KEY", "Treg API Key")
    .with_base_cost(150, BlockCostType.COST_USD)
    .build()
)

# The id the SDK gives the key it builds from TREG_API_KEY.
PLATFORM_CREDENTIALS_ID = "treg-default"


def is_platform_key(credentials: APIKeyCredentials) -> bool:
    """True for a Treg key the platform provides rather than the user's own."""
    return credentials.id == PLATFORM_CREDENTIALS_ID or is_system_credential(
        credentials.id
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="3f0b8a4e-6c1d-4f7a-9b2e-8d5c1a7e4f60",
    provider="treg",
    api_key=SecretStr("mock-treg-api-key"),
    title="Mock Treg API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
