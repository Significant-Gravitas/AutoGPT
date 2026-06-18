from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

DataForB2BCredentials = APIKeyCredentials
DataForB2BCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DATAFORB2B],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="dataforb2b",
    api_key=SecretStr("mock-dataforb2b-api-key"),
    title="Mock DataForB2B API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def DataForB2BCredentialsField() -> DataForB2BCredentialsInput:
    """Creates a DataForB2B credentials input on a block."""
    return CredentialsField(
        description="The DataForB2B integration can be used with an API Key.",
    )
