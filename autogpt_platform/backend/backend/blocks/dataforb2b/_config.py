"""Shared provider and credential configuration for DataForB2B blocks."""

from backend.sdk import (
    APIKeyCredentials,
    BlockCostType,
    CredentialsMetaInput,
    ProviderBuilder,
    ProviderName,
    SecretStr,
)

# DataForB2B is a metered, credit-consuming API and the "enrich_live" paths
# cost more credits upstream than the cached ones. 1 cr/RUN is a conservative
# interim floor that keeps every call billed, matching the agent_mail
# precedent. Revisit with per-endpoint / enrich_live-aware pricing once real
# DataForB2B usage costs are known.
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
TEST_CREDENTIALS_META_INPUT = DataForB2BCredentialsInput(
    provider=ProviderName("dataforb2b"),
    id=TEST_CREDENTIALS.id,
    type=TEST_CREDENTIALS.type,
    title=TEST_CREDENTIALS.title,
)
TEST_CREDENTIALS_INPUT = TEST_CREDENTIALS_META_INPUT.model_dump(mode="json")
