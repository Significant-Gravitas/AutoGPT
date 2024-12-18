from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

FalCredentials = APIKeyCredentials
FalCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.FAL],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="fal",
    api_key=SecretStr("mock-fal-api-key"),
    title="Mock FAL API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def FalCredentialsField() -> FalCredentialsInput:
    """
    Creates a FAL credentials input on a block.
    """
    return CredentialsField(
        description="The FAL integration can be used with an API Key.",
    )
