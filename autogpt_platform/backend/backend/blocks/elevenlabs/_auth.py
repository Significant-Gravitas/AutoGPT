"""
Authentication module for ElevenLabs API integration.

This module provides credential types and test credentials for the ElevenLabs API integration.
It defines the structure for API key credentials used to authenticate with the ElevenLabs API
and provides mock credentials for testing purposes.
"""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.integrations.providers import ProviderName

# Define the type of credentials input expected for ElevenLabs API
ElevenLabsCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.ELEVENLABS], Literal["api_key"]
]

# Mock credentials for testing ElevenLabs API integration
TEST_CREDENTIALS = APIKeyCredentials(
    id="f8274359-45c8-48b7-b54d-a1c8c09ac2e8",
    provider="elevenlabs",
    api_key=SecretStr("mock-elevenlabs-api-key"),
    title="Mock ElevenLabs API key",
    expires_at=None,
)

# Dictionary representation of test credentials for input fields
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
