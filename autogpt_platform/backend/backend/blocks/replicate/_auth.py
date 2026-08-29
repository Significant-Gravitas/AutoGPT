from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsMetaInput, ProviderName

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="replicate",
    api_key=SecretStr("mock-replicate-api-key"),
    title="Mock Replicate API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}

ReplicateCredentials = APIKeyCredentials
ReplicateCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.REPLICATE], Literal["api_key"]
]
