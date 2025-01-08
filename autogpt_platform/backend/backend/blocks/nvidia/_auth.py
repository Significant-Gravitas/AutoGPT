from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

NvidiaCredentials = APIKeyCredentials
NvidiaCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.NVIDIA],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="nvidia",
    api_key=SecretStr("mock-nvidia-api-key"),
    title="Mock Nvidia API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def NvidiaCredentialsField() -> NvidiaCredentialsInput:
    """Creates an Nvidia credentials input on a block."""
    return CredentialsField(description="The Nvidia integration requires an API Key.")
