"""
Authentication module for Example API integration.

This module provides credential types and test credentials for the Example API integration.
It defines the structure for API key credentials used to authenticate with the Example API
and provides mock credentials for testing purposes.
"""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.integrations.providers import ProviderName

# Define the type of credentials input expected for Example API
ExampleCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.EXAMPLE_PROVIDER], Literal["api_key"]
]


# Mock credentials for testing Example API integration
TEST_CREDENTIALS = APIKeyCredentials(
    id="9191c4f0-498f-4235-a79c-59c0e37454d4",
    provider="example-provider",
    api_key=SecretStr("mock-example-api-key"),
    title="Mock Example API key",
    expires_at=None,
)

# Dictionary representation of test credentials for input fields
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
