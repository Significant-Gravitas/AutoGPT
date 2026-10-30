"""nace API-key registration for Drex Max blocks."""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.sdk import BlockCostType, ProviderBuilder

nace = (
    ProviderBuilder("nace")
    .with_description("Drex Max: typed decisions on text and images")
    .with_supported_auth_types("api_key")
    # Keys are the user's own: nace bills them per input token, so these blocks
    # charge no platform credits.
    .with_base_cost(0, BlockCostType.RUN)
    .build()
)

NaceCredentials = APIKeyCredentials
NaceCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.NACE], Literal["api_key"]
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="2d3c8f4a-7b61-4e0a-9c25-6f1d0b8a3e57",
    provider="nace",
    api_key=SecretStr("mock-nace-api-key"),
    title="Mock nace API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def NaceCredentialsField() -> NaceCredentialsInput:
    return CredentialsField(
        description="Your nace API key. nace bills usage directly to your account."
    )
