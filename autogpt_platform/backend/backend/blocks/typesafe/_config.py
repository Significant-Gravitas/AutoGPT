"""TypeSafe API-key registration for Jev blocks."""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.sdk import BlockCostType, ProviderBuilder

typesafe = (
    ProviderBuilder("typesafe")
    .with_description("Jev: fast, typed judgments with probabilities")
    .with_supported_auth_types("api_key")
    # User-supplied keys are billed by TypeSafe; no provider price is available.
    .with_base_cost(0, BlockCostType.RUN)
    .build()
)

TypeSafeCredentials = APIKeyCredentials
TypeSafeCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.TYPESAFE], Literal["api_key"]
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="typesafe",
    api_key=SecretStr("mock-typesafe-api-key"),
    title="Mock TypeSafe API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def TypeSafeCredentialsField() -> TypeSafeCredentialsInput:
    return CredentialsField(
        description="Your TypeSafe API key. TypeSafe bills usage directly to your account."
    )
