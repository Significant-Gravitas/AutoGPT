"""
Authentication module for Enrichlayer API integration.

This module provides credential types and test credentials for the Enrichlayer API.
"""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.integrations.providers import ProviderName

# Define the type of credentials input expected for Enrichlayer API
EnrichlayerCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.ENRICHLAYER], Literal["api_key"]
]

# Mock credentials for testing Enrichlayer API integration
TEST_CREDENTIALS = APIKeyCredentials(
    id="1234a567-89bc-4def-ab12-3456cdef7890",
    provider="enrichlayer",
    api_key=SecretStr("mock-enrichlayer-api-key"),
    title="Mock Enrichlayer API key",
    expires_at=None,
)

# Dictionary representation of test credentials for input fields
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
