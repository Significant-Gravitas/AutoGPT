from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

WaveSpeedCredentials = APIKeyCredentials
WaveSpeedCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.WAVESPEED],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="wavespeed",
    api_key=SecretStr("mock-wavespeed-api-key"),
    title="Mock WaveSpeed API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def WaveSpeedCredentialsField() -> WaveSpeedCredentialsInput:
    """
    Creates a WaveSpeed credentials input on a block.
    """
    return CredentialsField(
        description="The WaveSpeed integration can be used with an API Key. "
        "You can obtain one from https://wavespeed.ai.",
    )
