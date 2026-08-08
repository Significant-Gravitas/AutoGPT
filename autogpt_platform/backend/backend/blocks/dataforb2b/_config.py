"""Shared provider and credential configuration for DataForB2B blocks."""

from backend.sdk import (
    APIKeyCredentials,
    BlockCostType,
    CredentialsMetaInput,
    ProviderBuilder,
    ProviderName,
    SecretStr,
)

# DataForB2B is a metered, credit-consuming API (blocks that "enrich_live" or
# search note this costs more credits upstream), but had no BLOCK_COSTS entry
# so the platform billed nothing for these calls. 1 cr/call is a conservative
# interim floor so no DataForB2B usage leaks past billing, matching the
# agent_mail precedent. Revisit with real per-endpoint costs if/when
# DataForB2B publishes usage-based pricing tiers.
dataforb2b = (
    ProviderBuilder("dataforb2b")
    .with_description(
        "B2B data API — search leads, enrich profiles (from a LinkedIn URL), find work "
        "emails and phone numbers for prospecting and outreach."
    )
    .with_api_key("DATAFORB2B_API_KEY", "DataForB2B API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

DataForB2BCredentials = APIKeyCredentials
DataForB2BCredentialsInput = CredentialsMetaInput

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="dataforb2b",
    api_key=SecretStr("mock-dataforb2b-api-key"),
    title="Mock DataForB2B API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = DataForB2BCredentialsInput(
    provider=ProviderName("dataforb2b"),
    id=TEST_CREDENTIALS.id,
    type=TEST_CREDENTIALS.type,
    title=TEST_CREDENTIALS.title,
)
