from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

HeyGenCredentials = APIKeyCredentials
HeyGenCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.HEYGEN],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="12345678-89ab-cdef-0123-456789abcdef",
    provider="heygen",
    api_key=SecretStr("mock-heygen-api-key"),
    title="Mock HeyGen API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def HeyGenCredentialsField() -> HeyGenCredentialsInput:
    """
    Creates a HeyGen credentials input on a block.
    """
    return CredentialsField(
        description="The HeyGen integration can be used with an API Key.",
    )
