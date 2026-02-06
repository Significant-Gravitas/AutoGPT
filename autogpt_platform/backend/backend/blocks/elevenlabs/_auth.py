"""ElevenLabs integration blocks - test credentials and shared utilities."""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.integrations.providers import ProviderName

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="elevenlabs",
    api_key=SecretStr("mock-elevenlabs-api-key"),
    title="Mock ElevenLabs API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

ElevenLabsCredentials = APIKeyCredentials
ElevenLabsCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.ELEVENLABS], Literal["api_key"]
]
