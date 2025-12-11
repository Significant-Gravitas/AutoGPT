"""
Instagram authentication module for AutoGPT Platform.

This module provides authentication utilities for Instagram blocks.
"""

# Authentication imports
from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput

# Instagram credentials type
InstagramCredentials = APIKeyCredentials

# Instagram credentials input type
InstagramCredentialsInput = CredentialsMetaInput[
    Literal["instagram"],
    Literal["api_key"],
]


def InstagramCredentialsField() -> InstagramCredentialsInput:
    """
    Creates a credentials field for Instagram authentication.

    Returns:
        InstagramCredentialsInput: Credentials field configuration
    """
    return CredentialsField(
        provider="instagram",
        supported_credential_types={"api_key"},
        description="Instagram credentials (username:password stored as API key)",
    )


# Mock credentials for unit testing only - NOT real credentials
# These are used by the test framework to verify block schemas
# Real credentials are provided by users through the platform's secure credential system
TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="instagram",
    api_key=SecretStr("test_username:test_password"),
    title="Mock Instagram Credentials",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
