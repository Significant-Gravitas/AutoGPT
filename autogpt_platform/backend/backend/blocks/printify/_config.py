"""Shared provider and credential configuration for Printify blocks."""

from backend.sdk import (
    APIKeyCredentials,
    BlockCostType,
    CredentialsMetaInput,
    ProviderBuilder,
    ProviderName,
    SecretStr,
)

# Printify issues personal access tokens from the account's Connections page;
# apps that act for other merchants need Printify's approval for OAuth, so a
# personal token is the sign-in a merchant can complete on their own.
printify = (
    ProviderBuilder("printify")
    .with_description("Print-on-demand products, mockups and publishing")
    .with_api_key("PRINTIFY_API_KEY", "Printify Personal Access Token")
    .with_api_key_url("https://printify.com/app/account/api")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

PrintifyCredentials = APIKeyCredentials
PrintifyCredentialsInput = CredentialsMetaInput

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="printify",
    api_key=SecretStr("mock-printify-token"),
    title="Mock Printify token",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = CredentialsMetaInput(
    provider=ProviderName("printify"),
    id=TEST_CREDENTIALS.id,
    type=TEST_CREDENTIALS.type,
    title=TEST_CREDENTIALS.title,
).model_dump(mode="json")
